# -*- coding: utf-8 -*-
import os
import os.path
import sys
from glob import glob
import subprocess as sp
import librosa
import numpy as np
import pandas as pd
from PIL import Image

import torch
import cv2
import tkinter as tk
from tkinter import filedialog
from synctoolbox.feature.pitch import audio_to_pitch_features
from synctoolbox.feature.chroma import pitch_to_chroma, quantize_chroma
from nnAudio import features


class Console(tk.Text):
    # console to output print functions to the GUI as feedback
    def __init__(self, *args, **kwargs):
        kwargs.update({"state": "normal"})
        tk.Text.__init__(self, *args, **kwargs)
        self.bind("<Destroy>", self.reset)
        self.old_stdout = sys.stdout
        sys.stdout = self

    def delete(self, *args, **kwargs):
        self.config(state="normal")
        self.delete(*args, **kwargs)
        self.config(state="disabled")

    def write(self, content):
        self.config(state="normal")
        self.insert("end", content)
        self.config(state="disabled")

    def reset(self, event):
        sys.stdout = self.old_stdout


class HiddenPrints:
    # hide the print functions (to hide prints from default nnAudio and synctoolbox packages
    # because it is a bit annoying)
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class duplicateFinder:
    """This class find duplicates of audio recordings within given set. Computes the chroma representation of audio
    files and store it as png image. Then, it computes 2D convolution between chroma representations. It slides one
    chroma along the second chroma to find a match. If so, it returns the paths of matching chromas, thus recordings.
    """

    def __init__(self, outputPath=None, outputPath_img=None, extensions=None, chroma_filepath=None,
                 soft_val=0.7, ref_path=None, mono=True, debug=False, method_name='cv2.TM_CCOEFF_NORMED',
                 chroma_method='cuda'):
        if extensions is None:
            self.extensions = ['aac', 'au', 'flac', 'm4a', 'mp3', 'ogg', 'wav']
        if outputPath is None:
            self.outputPath = 'df_data/output'
        else:
            self.outputPath = outputPath
        if outputPath_img is None:
            self.outputPath_img = 'df_data/img'
        else:
            self.outputPath_img = outputPath_img
        if chroma_filepath is None:
            self.chroma_filepath = 'df_data/chroma'
        else:
            self.chroma_filepath = chroma_filepath
        if ref_path is None:
            self.ref_path = 'df_data/reference'

        if not os.path.exists(self.outputPath):
            os.makedirs(self.outputPath)
        if not os.path.exists(self.outputPath_img):
            os.makedirs(self.outputPath_img)
        if not os.path.exists(self.chroma_filepath):
            os.makedirs(self.chroma_filepath)

        self.fs = 22050
        self.soft_val = soft_val
        self.filenames = []
        self.soft_filenames = []
        self.reference_filename = ''
        self.chroma_method = chroma_method
        self.mono = mono
        self.debug = debug
        self.diff_method_name = method_name
        self.diff_method = eval(self.diff_method_name)

    def run_on_files(self, path_to_files=None, input_files=None, reference=None, reference_name=None,
                     matching=False, matching_single=False, debug=False):
        # the main method to run the duplicate finder on audio files

        self.debug = debug

        if not path_to_files and not input_files:
            print('ERROR: Select a folder with files or separate audio files.')
            return None
        if matching and matching_single or (not matching and not matching_single):
            print("ERROR: Choose 'matching' or 'matching_single' parameter, not both or none.")
            return None
        if reference and reference_name:
            print('ERROR: Select a reference recording or give reference name, not both.')
            return None

        print(f'\nParameters selected:')
        if path_to_files:
            print(f'path_to_files: {path_to_files}')
        print(f'matching: {matching}')
        print(f'matching_single: {matching_single}')
        if reference:
            print(f'reference: {reference}')
        if reference_name:
            print(f'reference_name: {reference_name}')
        print(f'debug: {self.debug}')
        print(f'cuda is available: {torch.cuda.is_available()}')

        if path_to_files and input_files:
            partial_files = self.get_files(path_to_files=path_to_files)
            files = partial_files + input_files
        elif path_to_files:
            files = self.get_files(path_to_files=path_to_files)
        elif input_files:
            files = input_files.copy()
        else:
            print("ERROR: Choose a folder or separate recordings.")
            return None

        if matching_single:
            if reference == '' and reference_name == '':
                reference = files[0]
                files.remove(reference)
                if self.debug:
                    print('Reference chosen automatically (first item of the list).')
                    print(f'Reference: {reference}')
            elif reference_name != '':
                try:
                    reference = [file for file in files if reference_name in file][0]
                    print(f'Selected reference path: {reference}')
                    files.remove(reference)
                except:
                    print(f'ERROR: Could not find a reference recording by that name...')
                    return None
            else:
                reference = reference.replace("/", "\\")
                try:
                    files.remove(reference)
                    if self.debug:
                        print('Reference loaded successfully (included in the original list).')
                except:
                    if self.debug:
                        print('Reference is different from the target recordings.')

        self.create_dirs()

        self.get_chromagrams(files=files)
        self.export_chromaImages()

        if reference or reference_name:
            print(f'Total number of files: {len(self.filenames) + 1}\n')
        else:
            print(f'Total number of files: {len(self.filenames)}\n')

        if matching_single:
            hard_duplicates, soft_duplicates = self.return_single_matching_segments(files=files,
                                                                                    reference=reference,
                                                                                    method=self.diff_method,
                                                                                    debug=self.debug)
            hard_duplicates_dir = os.path.join(self.outputPath, 'dF_single_matching_duplicates' + '.xlsx')
            create_output_file(diff_structure_list=hard_duplicates, path_to_excel=hard_duplicates_dir)
            soft_duplicates_dir = os.path.join(self.outputPath, 'dF_single_matching_soft_duplicates' + '.xlsx')
            create_output_file(diff_structure_list=soft_duplicates, path_to_excel=soft_duplicates_dir)

        elif matching:
            hard_duplicates, soft_duplicates = self.return_matching_segments(files=files, filenames=self.filenames,
                                                                             method=self.diff_method, debug=self.debug)
            hard_duplicates_dir = os.path.join(self.outputPath, 'dF_matching_duplicates' + '.xlsx')
            create_output_file(diff_structure_list=hard_duplicates, path_to_excel=hard_duplicates_dir)
            soft_duplicates_dir = os.path.join(self.outputPath, 'dF_matching_soft_duplicates' + '.xlsx')
            create_output_file(diff_structure_list=soft_duplicates, path_to_excel=soft_duplicates_dir)

        else:
            print("ERROR: Choose 'matching_single' or 'matching'.")
            return None

        print('\n')
        if matching_single:
            print(f'Number of hard duplicates using single_matching: {len(hard_duplicates)}')
            print(f'Number of soft duplicates using single_matching: {len(soft_duplicates)}')
        if matching:
            print(f'Number of hard duplicates using matching: {len(hard_duplicates)}')
            print(f'Number of soft duplicates using matching: {len(soft_duplicates)}')
        print('\n')

        all_duplicates = hard_duplicates + soft_duplicates

        if all_duplicates:
            for duplicate in all_duplicates:
                print(f'Duplicate: {duplicate[0]} and {duplicate[1]}')
        print('\n')
        print('---DONE---')

    def create_dirs(self):
        # create directories if they do not exist
        dirs = [self.outputPath, self.outputPath_img, self.chroma_filepath, self.ref_path]
        for dir_path in dirs:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

    def get_files(self, path_to_files):
        # get path of all audio files with given extensions
        files = librosa.util.find_files(f'{path_to_files}/', ext=self.extensions)
        return files

    def get_chromagrams(self, files):
        # compute chroma based on synctoolbox or nnAudio based on chroma_method parameter; if cuda is available and
        # chroma_method == 'cuda', gpu is used and the computation is much faster
        filenames = []
        if self.debug:
            if torch.cuda.is_available():
                print('\n...using cuda...')
            else:
                print('\n...using cpu...')

        for file in files:
            filename = os.path.basename(file)
            filenames.append(filename)
            if not os.path.exists(f'{self.chroma_filepath}/{filename}.npy'):
                audio_data, _ = ffmpeg_load_audio(file, sr=self.fs, mono=self.mono)
                if self.chroma_method == 'cuda':
                    chroma = calculate_chromagram_cuda(audio_data)
                elif self.chroma_method == 'synctoolbox':
                    chroma = calculate_chromagram_synctoolbox(audio_data)
                else:
                    raise Exception('Choose the chroma method.')
                np.save(f'{self.chroma_filepath}/{filename}.npy', chroma)

        self.filenames = filenames

        print('\nChroma features have been successfuly extracted or loaded from all audio files')

    def export_chromaImages(self):
        # export the chroma representation as png image
        chroma_list = glob(self.chroma_filepath + '/*.npy')
        for chroma_file in chroma_list:
            filename = os.path.basename(chroma_file)[:-4]
            if not os.path.exists(f'{self.outputPath_img}/{filename}.png'):
                save_chromaImage(chroma_file, f'{self.outputPath_img}/{filename}.png')

        print("Saving chroma features as bitmaps for image processing...")

    def img_matching(self, img1_path, img2_path, soft_val=0.7, method=None, debug=False):
        # using matchTemplate and minMaxLoc functions to compare chromagrams of two recordings
        img = cv2.imread(img1_path, 0)
        template = cv2.imread(img2_path, 0)
        res = cv2.matchTemplate(img, template, method)
        _, max_val, _, _ = cv2.minMaxLoc(res)  # min_val, max_val, min_loc, max_loc
        if debug:
            print(f'max_val: {max_val}')

        if max_val >= 0.99:
            hard_duplicate = True
        else:
            hard_duplicate = False
        if soft_val <= max_val < 0.99:
            soft_duplicate = True
        else:
            soft_duplicate = False

        return hard_duplicate, soft_duplicate

    def return_matching_segments(self, files, filenames, method=eval('cv2.TM_CCOEFF_NORMED'), debug=False):
        # compare all recordings with all recordings to find duplicates
        name_pairs = []
        files_list = files.copy()
        files_number = len(files_list)
        filePairsList = []
        filenamesPairsList = []

        for i in range(0, files_number):
            for j in range(i + 1, files_number):
                filePairsList.append([files_list[i], files_list[j]])
                filenamesPairsList.append([filenames[i], filenames[j]])
        num_of_pairs = len(filePairsList)

        hard_duplicates = []
        soft_duplicates = []

        # Checking every pair if it is a duplicate
        for i, (filePair, filePairName) in enumerate(zip(filePairsList, filenamesPairsList), start=1):
            file1_name = filePairName[0]
            file2_name = filePairName[1]

            image1_dir = f'{self.outputPath_img}/{file1_name}.png'
            image2_dir = f'{self.outputPath_img}/{file2_name}.png'

            if self.debug:
                print(f'Checking if {file1_name} and {file2_name} are duplicates; '
                      f'[{str(i)}/{str(num_of_pairs)}]')

            is_duplicate_hard, is_duplicate_soft = self.img_matching(image1_dir, image2_dir, soft_val=self.soft_val,
                                                                     method=method, debug=debug)
            if is_duplicate_hard:
                hard_duplicates.append([filePair[0], filePair[1]])
            if is_duplicate_soft:
                soft_duplicates.append([filePair[0], filePair[1]])
            name_pairs.append(filePairName)

        return hard_duplicates, soft_duplicates

    def return_single_matching_segments(self, files, reference, method=eval('cv2.TM_CCOEFF_NORMED'), debug=False):
        # compare a reference audio file with all other files to find duplicates
        target_files = files.copy()
        files_number = len(target_files)
        hard_duplicates = []
        soft_duplicates = []

        ref_audio_data, _ = ffmpeg_load_audio(reference, sr=self.fs, mono=self.mono)
        ref_filename = os.path.basename(reference)

        if self.chroma_method == 'cuda':
            ref_chroma = calculate_chromagram_cuda(ref_audio_data)
        elif self.chroma_method == 'synctoolbox':
            ref_chroma = calculate_chromagram_synctoolbox(ref_audio_data)
        else:
            raise Exception('Chroma method has to be selected.')

        np.save(f'{self.ref_path}/{ref_filename}.npy', ref_chroma)
        save_chromaImage(f'{self.ref_path}/{ref_filename}.npy', f'{self.ref_path}/{ref_filename}.png')

        image1_dir = f'{self.ref_path}/{ref_filename}.png'

        # Checking every pair if it is a duplicate
        for i, file in enumerate(target_files, start=1):

            file2_name = os.path.basename(file)
            image2_dir = f'{self.outputPath_img}/{file2_name}.png'

            if self.debug:
                print(f'Checking if {ref_filename} and {file2_name} are duplicates; '
                      f'[{str(i)}/{str(files_number)}]')

            is_duplicate_hard, is_duplicate_soft = self.img_matching(image1_dir, image2_dir, soft_val=self.soft_val,
                                                                     method=method, debug=debug)
            if is_duplicate_hard:
                hard_duplicates.append([reference, file])
            if is_duplicate_soft:
                soft_duplicates.append([reference, file])

        return hard_duplicates, soft_duplicates


def create_output_file(diff_structure_list, path_to_excel):
    # create the excel output file
    df = pd.DataFrame(diff_structure_list)
    writer = pd.ExcelWriter(path_to_excel)
    df.to_excel(writer, sheet_name='duplicate_finder')
    writer.save()


def ffmpeg_load_audio(filename, sr=22050, mono=True, normalize=True, in_type=np.int16, out_type=np.float32,
                      DEVNULL=open(os.devnull, 'w')):
    # load an audio file via ffmpeg (needs to be in a SYSTEM path)
    channels = 1 if mono else 2
    format_strings = {
        np.float64: 'f64le',
        np.float32: 'f32le',
        np.int16: 's16le',
        np.int32: 's32le',
        np.uint32: 'u32le'
    }
    format_string = format_strings[in_type]
    command = [
        'ffmpeg',
        '-i', filename,
        '-f', format_string,
        '-acodec', 'pcm_' + format_string,
        '-ar', str(sr),
        '-ac', str(channels),
        '-']
    p = sp.Popen(command, stdout=sp.PIPE, stderr=DEVNULL, bufsize=4096, shell=True)
    bytes_per_sample = np.dtype(in_type).itemsize
    frame_size = bytes_per_sample * channels
    chunk_size = frame_size * sr  # read in 1-second chunks
    raw = b''
    with p.stdout as stdout:
        while True:
            data = stdout.read(chunk_size)
            if data:
                raw += data
            else:
                break
    # audio = np.fromstring(raw, dtype=in_type).astype(out_type) # older version
    audio = np.frombuffer(raw, dtype=in_type).astype(out_type)
    if channels > 1:
        audio = audio.reshape((-1, channels)).transpose()
    if audio.size == 0:
        return audio, sr
    if issubclass(out_type, np.floating):
        if normalize:
            peak = np.abs(audio).max()
            if peak > 0:
                audio /= peak
        elif issubclass(in_type, np.integer):
            audio /= np.iinfo(in_type).max
    return audio, sr


def calculate_chromagram_cuda(audio, fs=22050, spec_layer=None):
    # chroma features calculation using CUDA or CPU and nnAudio
    with HiddenPrints():
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if spec_layer is None:
            if torch.cuda.is_available():
                spec_layer = features.CQT(sr=fs, hop_length=512).cuda()
            else:
                spec_layer = features.CQT(sr=fs, hop_length=512).cpu()

        # creates cqt using nnaudio
        audio = torch.tensor(audio, device=device).float()
        cqt = spec_layer(audio)
        cqt = cqt.cpu().detach().numpy()[0]
        # calculates chromagram
        chroma = librosa.feature.chroma_cqt(C=cqt, sr=fs, hop_length=512)
    return chroma


def calculate_chromagram_synctoolbox(audio, fs=22050):
    # chroma features calculation using synctoolbox
    with HiddenPrints():
        f_pitch = audio_to_pitch_features(audio, Fs=fs)
        f_chroma = pitch_to_chroma(f_pitch=f_pitch)
        f_chroma_quantized = quantize_chroma(f_chroma=f_chroma)
    return f_chroma_quantized


def save_chromaImage(chroma_path, output_image):
    # saving chroma representation as png image
    chroma = np.load(chroma_path)
    chroma_processed = (chroma * 256).astype(np.uint8)
    im = Image.fromarray(chroma_processed)
    if im.mode != 'RGB':
        im = im.convert('RGB')
    im.save(output_image)


### functions for tkinter app ###
def browse_audio():
    # choose the folder with audio files
    dir_path = filedialog.askdirectory(title='Choose a directory')
    path_to_files.set(dir_path)
    startButton['state'] = 'normal'
    if dir_path:
        print(f'Chosen directory: {dir_path}')


def get_ref_name():
    # get the reference name
    input_text = text_box.get("1.0", "end-1c")
    global reference_name
    reference_name = input_text
    if reference_name:
        print(f'Reference name: {reference_name}')


def call_console():
    # call the console
    console_box = Console(root, height=30, width=100)
    console_box.grid(column=0, row=12, columnspan=10)
    scroll_bar = tk.Scrollbar(root, orient='vertical', command=console_box.yview)
    scroll_bar.grid(column=13, row=12, sticky=tk.NS)
    console_box['yscrollcommand'] = scroll_bar.set


def reset_selection():
    # reset all parameters and selected paths
    global path_to_files
    global input_files
    global reference
    global reference_name

    path_to_files = tk.StringVar()
    input_files = []
    reference = ''
    reference_name = ''
    text_box.delete('1.0', tk.END)
    c1.deselect()
    c2.deselect()
    c3.deselect()


def get_ref_filepath():
    # get a filepath of selected reference file
    file = filedialog.askopenfile(mode='r',
                                  filetypes=[('Audio Files', '*.wav .aac .au .flac .m4a .mp3 .ogg')],
                                  title='Choose a reference file')
    global reference
    if file:
        reference = os.path.abspath(file.name).replace('\\', '/')
        if reference:
            print(f'Chosen reference: {reference}')


def get_audio_files():
    # get filepath/s of selected files
    files = tk.filedialog.askopenfilenames(parent=root, title='Choose audio file/s')
    global input_files
    input_files += files
    input_files = list(set(input_files))  # removing duplicates
    startButton['state'] = 'normal'
    if files:
        print(f'Chosen recordings: {input_files}')


def quit(root):
    # destroy the window and python process
    root.destroy()


if __name__ == '__main__':
    # create the main window
    root = tk.Tk()
    root.title("Duplicate Finder")

    # definition of parameters
    path_to_files = tk.StringVar()
    input_files = []
    reference = ''
    reference_name = ''
    matching = tk.BooleanVar()
    matching_single = tk.BooleanVar()
    debug = tk.BooleanVar()

    # title and acknowledgment
    myLabel = tk.Label(root, text='Duplicate Finder', font="Calibri 14 bold")
    myLabel.grid(column=0, row=0, columnspan=11, padx=15, pady=15)
    authorLabel = tk.Label(root, text='git / xistva02', font="Helvetica 8")
    authorLabel.grid(column=9, row=0, sticky=tk.E)

    # button to choose the audio folder
    tk.Label(root, text='choose audio folder:').grid(column=0, row=2, sticky=tk.W)
    buttonBrowse = tk.Button(text="  ...  ", width=10, command=browse_audio)
    buttonBrowse.grid(column=1, row=2)

    # button to choose specific audio recordings
    tk.Label(root, text='or audio recordings:').grid(column=0, row=3, sticky=tk.W)
    buttonBrowse = tk.Button(text="  ...  ", width=10, command=get_audio_files)
    buttonBrowse.grid(column=1, row=3)

    # button to choose a reference recording (for single_matching)
    tk.Label(root, text='reference recording:').grid(column=0, row=4, sticky=tk.W)
    buttonBrowse = tk.Button(text=" ... ", width=10, command=get_ref_filepath)
    buttonBrowse.grid(column=1, row=4)

    # text widget to input a reference name (that corresponds to at least one of the selected recordings)
    displayLabel = tk.Label(root, text='or reference name:')
    displayLabel.grid(column=0, row=5, sticky=tk.W)
    text_box = tk.Text(root, height=1, width=20)
    text_box.grid(column=1, row=5, columnspan=3, sticky=tk.W)
    display_button = tk.Button(root, height=1, width=5, text="save", command=get_ref_name)
    display_button.grid(column=2, row=5)

    # boolean parameters to choose appropriate method for duplicate finder
    paramLabel = tk.Label(root, text='parameters', font="Calibri 10 bold")
    paramLabel.grid(column=0, row=6, columnspan=3, sticky=tk.W)
    c1 = tk.Checkbutton(root, text='matching', variable=matching, onvalue=1, offvalue=0)
    c1.grid(column=0, row=7)
    c2 = tk.Checkbutton(root, text='matching_single', variable=matching_single, onvalue=1, offvalue=0)
    c2.grid(column=1, row=7)
    c3 = tk.Checkbutton(root, text='debug', variable=debug, onvalue=1, offvalue=0)
    c3.grid(column=2, row=7)

    # button to clear the console
    clearLabel = tk.Label(root, text="clear console:")
    clearLabel.grid(column=0, row=8, sticky=tk.W)
    clearButton = tk.Button(root, height=1, text="clear", width=5, command=call_console)
    clearButton.grid(column=1, row=8)

    # button to reset the paths and parameters
    resetLabel = tk.Label(root, text="reset selection:")
    resetLabel.grid(column=0, row=9, sticky=tk.W)
    resetButton = tk.Button(root, height=1, text="reset", width=5, command=reset_selection)
    resetButton.grid(column=1, row=9)

    # button to start the analysis
    dF = duplicateFinder()
    startButton = tk.Button(root, text='Process', bg='#0073e6', fg='white',
                            command=lambda: [call_console(), print('...Processing...'),
                                             dF.run_on_files(path_to_files=path_to_files.get(),
                                                             input_files=input_files,
                                                             reference=reference,
                                                             reference_name=reference_name,
                                                             matching=matching.get(),
                                                             matching_single=matching_single.get(),
                                                             debug=debug.get())],
                            state=tk.DISABLED, height=2, width=15)
    startButton.grid(column=0, row=10, columnspan=11, sticky=tk.NS, padx=15, pady=15)

    # button to quit the window and python process
    quitButton = tk.Button(root, text="Quit", height=2, width=5, command=lambda: quit(root))
    quitButton.grid(column=9, row=10, sticky=tk.E, padx=15, pady=15)

    # call the console window for printing inside app
    call_console()

    root.mainloop()

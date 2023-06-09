﻿import os
import os.path
import sys
from glob import glob
import subprocess as sp
import librosa
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import warnings
import torch
import cv2
import tkinter as tk
from tkinter import filedialog
from synctoolbox.feature.pitch import audio_to_pitch_features
from synctoolbox.feature.chroma import pitch_to_chroma, quantize_chroma
from nnAudio import features

warnings.simplefilter("ignore", DeprecationWarning)


class Console(tk.Text):
    """
    Output print functions to the GUI 'console' as feedback.
    """
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

    def flush(self):
        pass


class HiddenPrints:
    """
    Class to hide the print functions (to suppress synctoolbox and nnAudio prints).
    """
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class duplicateFinder:
    def __init__(self,
                 output_path: str = 'df_data',
                 extensions: list = None,
                 soft_val: float = 0.5,
                 low_val: float = 0.05,
                 method_name: str = 'cv2.TM_CCOEFF_NORMED',
                 chroma_method: str = 'nnAudio'):
        """
        Compute duplicates within given audio recordings. Audio recordings are converted to chroma features
        via nnAudio module using cuda if available, else CPU. Chroma vectors are converted to grayscale images
        and saved. Matching method compares all input recordings with all input recordings, matching_single compares
        only one reference recording with the rest. Results are hard duplicates (exact match),
        soft duplicates (potential duplicate or similar structure), and low duplicates (files may be a bit similar).

        Parameters:
            output_path (str)           -- path to the directory where output chroma, images, and results will be stored
            extensions (list of str)    -- list of audio formats to look for
            soft_val (float)            -- soft value for duplicates
            low_val (float)             -- low value for duplicates
            method_name (str)           -- name of the method for features comparison (openCV)
            chroma_method (str)         -- chroma computation, nnAudio (optionally with cuda if available)
                                           or synctoolbox; synctoolbox is slower but should be more accurate
        """

        self.output_results = f'{output_path}/output'
        self.output_img = f'{output_path}/img'
        self.chroma_filepath = f'{output_path}/chroma'
        self.ref_path = f'{output_path}/reference'
        self.__fs = 22050
        self.output_path = output_path
        self.soft_val = soft_val
        self.low_val = low_val
        self.filenames = []
        self.filenames = []
        self.soft_filenames = []
        self.reference_filename = ''
        self.chroma_method = chroma_method
        self.mono = True
        self.debug = False
        self.diff_method_name = method_name
        self.diff_method = eval(self.diff_method_name)

        if extensions is None:
            self.extensions = ['aac', 'au', 'flac', 'm4a', 'mp3', 'ogg', 'wav']
        if not os.path.exists(self.output_results):
            os.makedirs(self.output_results)
        if not os.path.exists(self.output_img):
            os.makedirs(self.output_img)
        if not os.path.exists(self.chroma_filepath):
            os.makedirs(self.chroma_filepath)

    def update_dirs(self):
        self.output_results = f'{self.output_path}/output'
        self.output_img = f'{self.output_path}/img'
        self.chroma_filepath = f'{self.output_path}/chroma'
        self.ref_path = f'{self.output_path}/reference'

    def run_on_files(self,
                     path_to_files: str = None,
                     input_files: list = None,
                     reference: str = None,
                     reference_name: str = '',
                     output_path: str = 'df_data',
                     matching: bool = False,
                     matching_single: bool = False,
                     excel: bool = False,
                     csv: bool = True,
                     debug: bool = False):
        """
        Main method of the duplicateFinder class.

        Parameters:
            path_to_files (str)         -- path to the input files (subfolders are included)
            input_files (list of str)   -- list of strings of recordings (if different from path_to_files)
            reference (str)             -- path to the reference file if matching_single is True
            reference_name (str)        -- name of the reference file if inside of path_to_files
            output_path (str)           -- path to the output directory
            matching (str)              -- all vs. all comparison
            matching_single (str)       -- one vs. all comparison
            excel (bool)                -- if excel file should be outputted
            csv (bool)                  -- if csv file should be outputted
            debug (bool)                -- if True, additional info is outputted
        """

        if not path_to_files and not input_files:
            print("ERROR: Select a folder with files or separate audio files.")
            return None
        if matching and matching_single or (not matching and not matching_single):
            print("ERROR: Choose 'matching' or 'matching_single' parameter, not both or none.")
            return None
        if reference and reference_name:
            print("ERROR: Input a reference recording path or reference name, not both.")
            return None

        self.debug = debug
        self.output_path = output_path
        self.update_dirs()

        print(f'\nParameters selected:')
        if path_to_files:
            print(f'path_to_files: {path_to_files}')
        print(f'output_path: {self.output_path}')
        print(f'matching: {matching}')
        print(f'matching_single: {matching_single}')
        if reference:
            print(f'reference: {reference}')
        if reference_name:
            print(f'reference_name: {reference_name}')
        print(f'debug: {self.debug}')
        print(f'cuda is available: {torch.cuda.is_available()}')
        print(f'chroma_method: {self.chroma_method}\n')

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
                print('Reference chosen automatically (first item of the list).')
                print(f'Reference: {reference}')
            elif reference_name != '':
                try:
                    reference = [file for file in files if reference_name in file][0]
                    print(f'Selected reference path: {reference}')
                    files.remove(reference)
                except:
                    print(f"ERROR: Could not find a reference recording by that name...")
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
        self.export_chroma_images()

        if reference or reference_name:
            print(f'Total number of files: {len(self.filenames) + 1}\n')
        else:
            print(f'Total number of files: {len(self.filenames)}\n')

        if matching_single:
            method_used = 'matching_single'
            hard_duplicates, soft_duplicates, low_duplicates, result_all = self.return_single_matching_segments(
                files=files,
                reference=reference,
                method=self.diff_method,
                debug=self.debug)
        elif matching:
            method_used = 'matching'
            hard_duplicates, soft_duplicates, low_duplicates, result_all = self.return_matching_segments(
                files=files, filenames=self.filenames,
                method=self.diff_method, debug=self.debug)
        else:
            print("ERROR: Choose 'matching_single' or 'matching'.")
            return None

        hard_duplicates.sort()
        soft_duplicates.sort()
        low_duplicates.sort()
        result_all.sort()

        self.create_output_file(duplicates=hard_duplicates,
                                path_to_file=f'{self.output_results}/dF_{method_used}_hard_duplicates',
                                excel=excel, csv=csv)
        self.create_output_file(duplicates=soft_duplicates,
                                path_to_file=f'{self.output_results}/dF_{method_used}_soft_duplicates',
                                excel=excel, csv=csv)
        self.create_output_file(duplicates=low_duplicates,
                                path_to_file=f'{self.output_results}/dF_{method_used}_low_duplicates',
                                excel=excel, csv=csv)
        self.create_output_file(duplicates=result_all,
                                path_to_file=f'{self.output_results}/dF_all_files',
                                excel=excel, csv=csv)

        if matching_single:
            print(f'Number of hard duplicates using single_matching: {len(hard_duplicates)}')
            print(f'Number of soft duplicates using single_matching: {len(soft_duplicates)}')
            print(f'Number of low duplicates using single_matching: {len(low_duplicates)}')
        if matching:
            print(f'Number of hard duplicates using matching: {len(hard_duplicates)}')
            print(f'Number of soft duplicates using matching: {len(soft_duplicates)}')
            print(f'Number of low duplicates using single_matching: {len(low_duplicates)}')

        all_duplicates = hard_duplicates + soft_duplicates

        if all_duplicates:
            print('\n')
            for duplicate in all_duplicates:
                print(f'Duplicate: {duplicate[1]} and {duplicate[3]}')
        print('\n')
        print('---DONE---')

    def return_matching_segments(self,
                                 files: list,
                                 filenames: list,
                                 method=eval('cv2.TM_CCOEFF_NORMED'),
                                 debug: bool = False):
        """
        Load and compare all images and return duplicates and all combinations.
        """
        files_list = files.copy()
        files_number = len(files_list)
        file_pairs_list = []
        filenames_pairs_list = []

        for i in range(0, files_number):
            for j in range(i + 1, files_number):
                file_pairs_list.append([files_list[i], files_list[j]])
                filenames_pairs_list.append([filenames[i], filenames[j]])
        num_of_pairs = len(file_pairs_list)

        print(f'Total number of combinations: {str(num_of_pairs)}')

        hard_duplicates = []
        soft_duplicates = []
        low_duplicates = []
        result_all = []

        # Checking every pair if it is a duplicate
        for i, (filePair, filePairName) in enumerate(zip(file_pairs_list, filenames_pairs_list), start=1):
            file1_name = filePairName[0]
            file2_name = filePairName[1]

            image1_dir = f'{self.output_img}/{file1_name}.png'
            image2_dir = f'{self.output_img}/{file2_name}.png'

            is_duplicate_hard, is_duplicate_soft, is_low_duplicate, result_value = self.img_matching(image1_dir,
                                                                                                     image2_dir,
                                                                                                     soft_val=self.soft_val,
                                                                                                     low_val=self.low_val,
                                                                                                     method=method,
                                                                                                     debug=debug)
            if self.debug:
                print(f'Checking if {file1_name} and {file2_name} are duplicates; [{str(i)}/{str(num_of_pairs)}]: '
                      f'max_val: {result_value}')

            if is_duplicate_hard:
                hard_duplicates.append([filePair[0], file1_name, filePair[1], file2_name, result_value])
            if is_duplicate_soft:
                soft_duplicates.append([filePair[0], file1_name, filePair[1], file2_name, result_value])
            if is_low_duplicate:
                low_duplicates.append([filePair[0], file1_name, filePair[1], file2_name, result_value])

            result_all.append([filePair[0], file1_name, filePair[1], file2_name, result_value])

        return hard_duplicates, soft_duplicates, low_duplicates, result_all

    def return_single_matching_segments(self,
                                        files: list,
                                        reference: str,
                                        method: str = eval('cv2.TM_CCOEFF_NORMED'),
                                        debug: bool = False):
        """
        matching_single method (one vs. all comparison)
        Data for the reference recordings are stored separately.
        """
        target_files = files.copy()
        files_number = len(target_files)
        hard_duplicates = []
        soft_duplicates = []
        low_duplicates = []
        result_all = []

        with HiddenPrints():
            ref_audio_data, _ = self.ffmpeg_load_audio(filename=reference, sr=self.__fs, mono=self.mono)        
            ref_filename = os.path.basename(reference)
            if self.chroma_method == 'nnAudio':
                ref_chroma = self.calculate_chromagram_nnAudio(ref_audio_data)
            elif self.chroma_method == 'synctoolbox':
                ref_chroma = self.calculate_chromagram_synctoolbox(ref_audio_data)
            else:
                print('Chroma method has to be selected.')
                return None

        np.save(f'{self.ref_path}/{ref_filename}.npy', ref_chroma)
        self.save_chroma_image(f'{self.ref_path}/{ref_filename}.npy', f'{self.ref_path}/{ref_filename}.png')

        image1_dir = f'{self.ref_path}/{ref_filename}.png'

        # Checking every pair if it is a duplicate
        for i, file in enumerate(target_files, start=1):

            file2_name = os.path.basename(file)
            image2_dir = f'{self.output_img}/{file2_name}.png'

            is_duplicate_hard, is_duplicate_soft, is_low_duplicate, result_value = self.img_matching(image1_dir,
                                                                                                     image2_dir,
                                                                                                     soft_val=self.soft_val,
                                                                                                     low_val=self.low_val,
                                                                                                     method=method,
                                                                                                     debug=debug)

            if self.debug:
                print(f'Checking if {ref_filename} and {file2_name} are duplicates; [{str(i)}/{str(files_number)}]: '
                      f'max_val: {result_value}')

            if is_duplicate_hard:
                hard_duplicates.append([reference, ref_filename, file, file2_name, result_value])
            if is_duplicate_soft:
                soft_duplicates.append([reference, ref_filename, file, file2_name, result_value])
            if is_low_duplicate:
                low_duplicates.append([reference, ref_filename, file, file2_name, result_value])

            result_all.append([reference, ref_filename, file, file2_name, result_value])

        return hard_duplicates, soft_duplicates, low_duplicates, result_all

    # helper functions
    def create_dirs(self):
        """
        Create all required dirs: results, images, chromas and ref data for matching_single.
        """
        dirs = [self.output_results, self.output_img, self.chroma_filepath, self.ref_path]
        for dir_path in dirs:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

    def get_files(self,
                  path_to_files: str):
        """
        Get all files with selected extensions from path_to_files path and all subdirectories
        """
        files = librosa.util.find_files(path_to_files + '/', ext=self.extensions)
        return files

    def get_chromagrams(self,
                        files: list):
        """
        Extract chromagrams using either nnAudio or synctoolbox.
        """
        filenames = []
        print(f'...extracting chroma features...')
        with HiddenPrints():
            for file in files:
                filename = os.path.basename(file)
                filenames.append(filename)
                if not os.path.exists(f'{self.chroma_filepath}/{filename}.npy'):
                    audio_data, _ = self.ffmpeg_load_audio(filename=file, sr=self.__fs, mono=self.mono)
                    if self.chroma_method == 'nnAudio':
                        chroma = self.calculate_chromagram_nnAudio(audio_data)
                    elif self.chroma_method == 'synctoolbox':
                        chroma = self.calculate_chromagram_synctoolbox(audio_data)
                    else:
                        print('Choose the chroma method.')
                    np.save(f'{self.chroma_filepath}/{filename}.npy', chroma)
        self.filenames = filenames

        print('\nChroma features have been successfully extracted or loaded from all audio files')

    def export_chroma_images(self):
        """
        Export chroma as .png images.
        """
        chroma_list = glob(self.chroma_filepath + '/*.npy')
        for chroma_file in chroma_list:
            filename = os.path.basename(chroma_file)[:-4]
            if not os.path.exists(f'{self.output_img}/{filename}.png'):
                self.save_chroma_image(chroma_file, f'{self.output_img}/{filename}.png')

        print('...saving chroma features as bitmaps for image processing...')

    @staticmethod
    def img_matching(img1_path: str,
                     img2_path: str,
                     soft_val: float = 0.5,
                     low_val: float = 0.05,
                     debug: bool = False,
                     method=eval('cv2.TM_CCOEFF_NORMED')):
        """
        Load images and compare them using matchTemplate and minMaxLoc methods.
        It returns hard, soft, and low duplicates and the value (between 0 and 1) of maximum 'similarity'.
        """
        img = cv2.imread(img1_path, 0)
        template = cv2.imread(img2_path, 0)
        res = cv2.matchTemplate(img, template, method)
        _, max_val, _, _ = cv2.minMaxLoc(res)  # min_val, max_val, min_loc, max_loc
        max_val = round(max_val, 3)

        hard_duplicate = False
        soft_duplicate = False
        low_duplicate = False

        if max_val >= 0.9:
            hard_duplicate = True
        if soft_val <= max_val < 0.9:
            soft_duplicate = True
        if low_val <= max_val < soft_val:
            low_duplicate = True

        return hard_duplicate, soft_duplicate, low_duplicate, max_val

    @staticmethod
    def create_output_file(duplicates: list,
                           path_to_file: str,
                           remove_previous: bool = False,
                           excel: bool = True,
                           csv: bool = True):
        """
        Save the duplicates in the .xlsx or .csv file (or both).
        """
        if remove_previous:
            if os.path.exists(path_to_file + '.xlsx'):
                os.remove(path_to_file + '.xlsx')
        columns = ['ref_path', 'filename1', 'target_path', 'filename2', 'max_value']
        df = pd.DataFrame(duplicates, columns=columns)
        if excel:
            try:
                writer = pd.ExcelWriter(path_to_file + '.xlsx')
                df.to_excel(writer, sheet_name='duplicate_finder', index=False)
                writer.save()
            except:
                print('Cannot write the excel file...')
        if csv:
            df.to_csv(path_to_file + '.csv', index=False)

    @staticmethod
    def ffmpeg_load_audio(filename: str,
                          sr: int = 22050,
                          mono: bool = True):
        """
        Ffmpeg function for the non .wav audio files.        
        """
        channels = 1 if mono else 2
        command = [
            'ffmpeg',
            '-i', filename,
            '-f', 's16le',
            '-acodec', 'pcm_s16le',
            '-ar', str(sr),
            '-ac', str(channels),
            '-hide_banner',
            '-loglevel', 'warning',
            '-']
        
        pipe = sp.Popen(command, stdout=sp.PIPE)
        stdoutdata = pipe.stdout.read()
        audio_array = np.fromstring(stdoutdata, dtype="int16")

        if channels == 2:
            audio_array = audio_array.reshape((len(audio_array)/2,2))
        return audio_array, sr



    # @staticmethod
    # def ffmpeg_load_audio_full(filename: str,
    #                       sr: int = 22050,
    #                       mono: bool = True,
    #                       normalize: bool = True,
    #                       in_type=np.int16,
    #                       out_type=np.float32,
    #                       DEVNULL=open(os.devnull, 'w')):
    #     """
    #     Ffmpeg function for the non .wav audio files. Enables full control of the audio data.
    #     In this setting, it works only on Windows.
    #     """
    #     channels = 1 if mono else 2
    #     format_strings = {
    #         np.float64: 'f64le',
    #         np.float32: 'f32le',
    #         np.int16: 's16le',
    #         np.int32: 's32le',
    #         np.uint32: 'u32le'
    #     }
    #     format_string = format_strings[in_type]
    #     command = [
    #         'ffmpeg',
    #         '-i', filename,
    #         '-f', format_string,
    #         '-acodec', 'pcm_' + format_string,
    #         '-ar', str(sr),
    #         '-ac', str(channels),
    #         '-']
    #     p = sp.Popen(command, stdout=sp.PIPE, stderr=DEVNULL, bufsize=4096, shell=True)
    #     bytes_per_sample = np.dtype(in_type).itemsize
    #     frame_size = bytes_per_sample * channels
    #     chunk_size = frame_size * sr  # read in 1-second chunks
    #     raw = b''
    #     with p.stdout as stdout:
    #         while True:
    #             data = stdout.read(chunk_size)
    #             if data:
    #                 raw += data
    #             else:
    #                 break
    #     # audio = np.fromstring(raw, dtype=in_type).astype(out_type) # older version
    #     audio = np.frombuffer(raw, dtype=in_type).astype(out_type)
    #     if channels > 1:
    #         audio = audio.reshape((-1, channels)).transpose()
    #     if audio.size == 0:
    #         return audio, sr
    #     if issubclass(out_type, np.floating):
    #         if normalize:
    #             peak = np.abs(audio).max()
    #             if peak > 0:
    #                 audio /= peak
    #         elif issubclass(in_type, np.integer):
    #             audio /= np.iinfo(in_type).max
    #     return audio, sr

    @staticmethod
    def calculate_chromagram_nnAudio(audio: np.ndarray,
                                     fs: int = 22050,
                                     spec_layer=None):
        """
        Calculate chroma features using nnAudio module (cuda or cpu).
        """
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

    @staticmethod
    def calculate_chromagram_synctoolbox(audio: np.ndarray,
                                         fs: int = 22050):
        """
        Calculate chroma features using synctoolbox module (cpu).
        """
        f_pitch = audio_to_pitch_features(audio, Fs=fs)
        f_chroma = pitch_to_chroma(f_pitch=f_pitch)
        f_chroma_quantized = quantize_chroma(f_chroma=f_chroma)
        return f_chroma_quantized

    @staticmethod
    def save_chroma_image(chroma_path: str,
                          output_image: str):
        """
        Save the chroma image as .png format.
        """
        chroma = np.load(chroma_path)
        chroma_processed = (chroma * 256).astype(np.uint8)
        im = Image.fromarray(chroma_processed)
        gray_im = ImageOps.grayscale(im)
        # if im.mode != 'RGB':
        #     im = im.convert('RGB')
        gray_im.save(output_image)


### functions for tkinter app ###
def browse_audio():
    """
    Choose the folder with audio files.
    """
    dir_path = filedialog.askdirectory(title='Choose a directory')
    path_to_files.set(dir_path)
    startButton['state'] = 'normal'
    if dir_path:
        print(f'Chosen input directory: {dir_path}')

def get_output_dir():
    """
    Choose the output folder.
    """
    out_path = filedialog.askdirectory(title='Choose a directory')
    output_path.set(out_path)
    startButton['state'] = 'normal'
    if out_path:
        print(f'Chosen output directory: {out_path}')


def get_ref_name():
    """
    Get the reference name.
    """
    input_text = text_box.get("1.0", "end-1c")
    global reference_name
    reference_name = input_text
    if reference_name:
        print(f'Reference name: {reference_name}')


def call_console():
    """
    Call the console.
    """
    console_box = Console(root, height=30, width=100)
    console_box.grid(column=0, row=15, columnspan=10)
    scroll_bar = tk.Scrollbar(root, orient='vertical', command=console_box.yview)
    scroll_bar.grid(column=13, row=15, sticky=tk.NS)
    console_box['yscrollcommand'] = scroll_bar.set


def reset_selection():
    """
    Reset all parameters and selected paths.
    """
    global path_to_files
    global input_files
    global output_path
    global reference
    global reference_name

    path_to_files = tk.StringVar()
    output_path = tk.StringVar(value='df_data')
    input_files = []
    reference = ''
    reference_name = ''
    text_box.delete('1.0', tk.END)
    c1.deselect()
    c2.deselect()
    c3.deselect()


def get_ref_filepath():
    """
    Get a filepath of selected reference file.
    """
    file = filedialog.askopenfile(mode='r',
                                  filetypes=[('Audio Files', '*.wav .aac .au .flac .m4a .mp3 .ogg')],
                                  title='Choose a reference file')
    global reference
    if file:
        reference = os.path.abspath(file.name).replace('\\', '/')
        if reference:
            print(f'Chosen reference: {reference}')


def get_audio_files():
    """
    Get filepath/s of selected files.
    """
    files = tk.filedialog.askopenfilenames(parent=root, title='Choose audio file/s')
    global input_files
    input_files += files
    input_files = list(set(input_files))  # removing duplicates
    startButton['state'] = 'normal'
    if files:
        print(f'Chosen input recordings: {input_files}')


def quit(root):
    """
    Destroy the window and python process.
    """
    root.destroy()


if __name__ == '__main__':
    # create the main window
    root = tk.Tk()
    root.title("Duplicate Finder")

    # definition of parameters
    path_to_files = tk.StringVar()
    output_path = tk.StringVar(value='df_data')
    input_files = []
    reference = ''
    reference_name = ''
    matching = tk.BooleanVar(value=True)
    matching_single = tk.BooleanVar()
    excel = tk.BooleanVar(value=False)
    csv = tk.BooleanVar(value=True)
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

    # button to choose the output folder
    tk.Label(root, text='choose output folder:').grid(column=0, row=6, sticky=tk.W)
    buttonBrowse = tk.Button(text="  ...  ", width=10, command=get_output_dir)
    buttonBrowse.grid(column=1, row=6)

    # boolean parameters to choose appropriate method for duplicate finder
    paramLabel = tk.Label(root, text='parameters:', font="Calibri 10 bold")
    paramLabel.grid(column=0, row=7, columnspan=3, sticky=tk.W)
    c1 = tk.Checkbutton(root, text='matching', variable=matching, onvalue=1, offvalue=0)
    c1.grid(column=0, row=8)
    c2 = tk.Checkbutton(root, text='matching_single', variable=matching_single, onvalue=1, offvalue=0)
    c2.grid(column=1, row=8)
    c3 = tk.Checkbutton(root, text='debug', variable=debug, onvalue=1, offvalue=0)
    c3.grid(column=2, row=8)

    # boolean parameters to choose the .csv or .xlsx outputs
    paramLabel = tk.Label(root, text='output files:', font="Calibri 10 bold")
    paramLabel.grid(column=0, row=9, columnspan=3, sticky=tk.W)
    c1 = tk.Checkbutton(root, text='.xlsx', variable=excel, onvalue=1, offvalue=0)
    c1.grid(column=0, row=10)
    c2 = tk.Checkbutton(root, text='.csv', variable=csv, onvalue=1, offvalue=0)
    c2.grid(column=1, row=10)


    # button to clear the console
    clearLabel = tk.Label(root, text="clear console:")
    clearLabel.grid(column=0, row=11, sticky=tk.W)
    clearButton = tk.Button(root, height=1, text="clear", width=5, command=call_console)
    clearButton.grid(column=1, row=11)

    # button to reset the paths and parameters
    resetLabel = tk.Label(root, text="reset selection:")
    resetLabel.grid(column=0, row=12, sticky=tk.W)
    resetButton = tk.Button(root, height=1, text="reset", width=5, command=reset_selection)
    resetButton.grid(column=1, row=12)

    # button to start the analysis
    dF = duplicateFinder()
    startButton = tk.Button(root, text='Process', bg='#0073e6', fg='white',
                            command=lambda: [call_console(), print('...processing...'),
                                             dF.run_on_files(path_to_files=path_to_files.get(),
                                                             input_files=input_files,
                                                             reference=reference,
                                                             reference_name=reference_name,
                                                             output_path=output_path.get(),
                                                             matching=matching.get(),
                                                             matching_single=matching_single.get(),
                                                             excel=excel.get(),
                                                             csv=csv.get(),
                                                             debug=debug.get())],
                            state=tk.DISABLED, height=2, width=15)
    startButton.grid(column=0, row=13, columnspan=11, sticky=tk.NS, padx=15, pady=15)

    # button to quit the window and python process
    quitButton = tk.Button(root, text="Quit", height=2, width=5, command=lambda: quit(root))
    quitButton.grid(column=9, row=13, sticky=tk.E, padx=15, pady=15)

    # call the console window for printing inside app
    call_console()

    root.mainloop()

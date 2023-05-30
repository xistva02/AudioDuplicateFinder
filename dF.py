import os
import os.path
from glob import glob
import subprocess as sp
import librosa
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from argparse import ArgumentParser
import torch
import cv2
from synctoolbox.feature.pitch import audio_to_pitch_features
from synctoolbox.feature.chroma import pitch_to_chroma, quantize_chroma
from nnAudio import features


class duplicateFinder:
    def __init__(self,
                 output_path: str = 'df_data',
                 extensions: list = None,
                 soft_val: float = 0.5,
                 low_val: float = 0.05,
                 method_name: str = 'cv2.TM_CCOEFF_NORMED',
                 chroma_method: str = 'cuda',
                 debug: bool = False):
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
            chroma_method (str)         -- chroma computation, cuda or CPU
            debug (boolean)             -- if True, additional information is printed
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
        self.soft_filenames = []
        self.reference_filename = ''
        self.chroma_method = chroma_method
        self.mono = True
        self.debug = debug
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

    def run_on_files(self,
                     path_to_files: str = None,
                     input_files: list = None,
                     reference: str = None,
                     reference_name: str = None,
                     matching: bool = False,
                     matching_single: bool = False):
        """
        Main method of the duplicateFinder class.

        Parameters:
            path_to_files (str)         -- path to the input files (subfolders are included)
            input_files (list of str)   -- list of strings of recordings (if different from path_to_files)
            reference (str)             -- path to the reference file if matching_single is True
            reference_name (str)        -- name of the reference file if inside of path_to_files
            matching (str)              -- all vs. all comparison
            matching_single (str)       -- one vs. all comparison
        """

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
        print(f'output_path = {self.output_path}')
        print(f'matching: {matching}')
        print(f'matching_single: {matching_single}')
        if reference:
            print(f'reference: {reference}')
        if reference_name:
            print(f'reference_name: {reference_name}')
        print(f'debug: {self.debug}')
        print(f'cuda is available: {torch.cuda.is_available()}\n')

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
                                path_to_file=f'{self.output_results}/dF_{method_used}_hard_duplicates')
        self.create_output_file(duplicates=soft_duplicates,
                                path_to_file=f'{self.output_results}/dF_{method_used}_soft_duplicates')
        self.create_output_file(duplicates=low_duplicates,
                                path_to_file=f'{self.output_results}/dF_{method_used}_low_duplicates')
        self.create_output_file(duplicates=result_all,
                                path_to_file=f'{self.output_results}/dF_all_files')

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

    def create_dirs(self):
        dirs = [self.output_results, self.output_img, self.chroma_filepath, self.ref_path]
        for dir_path in dirs:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

    def update_path_to_files(self,
                             path_to_files: str):
        self.path_to_files = path_to_files

    def get_files(self,
                  path_to_files: str):
        files = librosa.util.find_files(path_to_files + '/', ext=self.extensions)
        return files

    def get_chromagrams(self,
                        files: list):
        filenames = []
        print(f'extracting chroma features...')
        for file in tqdm(files):
            filename = os.path.basename(file)
            filenames.append(filename)
            if not os.path.exists(f'{self.chroma_filepath}/{filename}.npy'):
                audio_data, _ = self.ffmpeg_load_audio(file, sr=self.__fs, mono=self.mono)
                if self.chroma_method == 'cuda':
                    chroma = self.calculate_chromagram_cuda(audio_data)
                elif self.chroma_method == 'synctoolbox':
                    chroma = self.calculate_chromagram_synctoolbox(audio_data)
                else:
                    raise Exception('Choose the chroma method.')
                np.save(f'{self.chroma_filepath}/{filename}.npy', chroma)
        self.filenames = filenames

        print('\nChroma features have been successfully extracted or loaded from all audio files')

    def export_chromaImages(self):
        chroma_list = glob(self.chroma_filepath + '/*.npy')
        for chroma_file in chroma_list:
            filename = os.path.basename(chroma_file)[:-4]
            if not os.path.exists(f'{self.output_img}/{filename}.png'):
                self.save_chromaImage(chroma_file, f'{self.output_img}/{filename}.png')

        print('saving chroma features as bitmaps for image processing...')

    @staticmethod
    def img_matching(img1_path: str,
                     img2_path: str,
                     soft_val: float = 0.5,
                     low_val: float = 0.05,
                     debug: bool = False,
                     method=eval('cv2.TM_CCOEFF_NORMED')):
        # load grayscale images of chroma
        img = cv2.imread(img1_path, 0)
        template = cv2.imread(img2_path, 0)
        res = cv2.matchTemplate(img, template, method)
        _, max_val, _, _ = cv2.minMaxLoc(res)  # min_val, max_val, min_loc, max_loc
        if debug:
            print(f'max_val: {max_val}')

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

    def return_matching_segments(self,
                                 files: list,
                                 filenames: list,
                                 method=eval('cv2.TM_CCOEFF_NORMED'),
                                 debug: bool = False):
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

            if self.debug:
                print(f'Checking if {file1_name} and {file2_name} are duplicates; '
                      f'[{str(i)}/{str(num_of_pairs)}]')

            is_duplicate_hard, is_duplicate_soft, is_low_duplicate, result_value = self.img_matching(image1_dir,
                                                                                                     image2_dir,
                                                                                                     soft_val=self.soft_val,
                                                                                                     low_val=self.low_val,
                                                                                                     method=method,
                                                                                                     debug=debug)
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
        target_files = files.copy()
        files_number = len(target_files)
        hard_duplicates = []
        soft_duplicates = []
        low_duplicates = []
        result_all = []

        ref_audio_data, _ = self.ffmpeg_load_audio(reference, sr=self.__fs, mono=self.mono)
        ref_filename = os.path.basename(reference)
        if self.chroma_method == 'cuda':
            ref_chroma = self.calculate_chromagram_cuda(ref_audio_data)
        elif self.chroma_method == 'synctoolbox':
            ref_chroma = self.calculate_chromagram_synctoolbox(ref_audio_data)
        else:
            raise Exception('Chroma method has to be selected.')

        np.save(f'{self.ref_path}/{ref_filename}.npy', ref_chroma)
        self.save_chromaImage(f'{self.ref_path}/{ref_filename}.npy', f'{self.ref_path}/{ref_filename}.png')

        image1_dir = f'{self.ref_path}/{ref_filename}.png'

        # Checking every pair if it is a duplicate
        for i, file in enumerate(target_files, start=1):

            file2_name = os.path.basename(file)
            image2_dir = f'{self.output_img}/{file2_name}.png'

            if self.debug:
                print(f'Checking if {ref_filename} and {file2_name} are duplicates; '
                      f'[{str(i)}/{str(files_number)}]')

            is_duplicate_hard, is_duplicate_soft, is_low_duplicate, result_value = self.img_matching(image1_dir,
                                                                                                     image2_dir,
                                                                                                     soft_val=self.soft_val,
                                                                                                     low_val=self.low_val,
                                                                                                     method=method,
                                                                                                     debug=debug)
            if is_duplicate_hard:
                hard_duplicates.append([reference, ref_filename, file, file2_name, result_value])
            if is_duplicate_soft:
                soft_duplicates.append([reference, ref_filename, file, file2_name, result_value])
            if is_low_duplicate:
                low_duplicates.append([reference, ref_filename, file, file2_name, result_value])

            result_all.append([reference, ref_filename, file, file2_name, result_value])

        return hard_duplicates, soft_duplicates, low_duplicates, result_all

    @staticmethod
    def create_output_file(duplicates: list,
                           path_to_file: str,
                           remove_previous: bool = False):
        if remove_previous:
            if os.path.exists(path_to_file + '.xlsx'):
                os.remove(path_to_file + '.xlsx')
        columns = ['ref_path', 'filename1', 'target_path', 'filename2', 'max_value']
        df = pd.DataFrame(duplicates, columns=columns)
        try:
            writer = pd.ExcelWriter(path_to_file + '.xlsx')
            df.to_excel(writer, sheet_name='duplicate_finder')
            writer.save()
        except:
            print('Cannot write the excel file...')
        df.to_csv(path_to_file + '.csv')

    @staticmethod
    def ffmpeg_load_audio(filename: str,
                          sr: int = 22050,
                          mono: bool = True,
                          normalize: bool = True,
                          in_type=np.int16,
                          out_type=np.float32,
                          DEVNULL=open(os.devnull, 'w')):
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

    @staticmethod
    # function for chroma features calculation using CUDA or CPU and nnAudio
    def calculate_chromagram_cuda(audio: np.ndarray,
                                  fs: int = 22050,
                                  spec_layer=None):
        # initializes spectrogram layer
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
    # function for chroma features calculation using synctoolbox
    def calculate_chromagram_synctoolbox(audio: np.ndarray,
                                         fs: int = 22050):
        f_pitch = audio_to_pitch_features(audio, Fs=fs)
        f_chroma = pitch_to_chroma(f_pitch=f_pitch)
        f_chroma_quantized = quantize_chroma(f_chroma=f_chroma)
        return f_chroma_quantized

    @staticmethod
    def save_chromaImage(chroma_path: str,
                         output_image: str):
        chroma = np.load(chroma_path)
        chroma_processed = (chroma * 256).astype(np.uint8)
        im = Image.fromarray(chroma_processed)
        if im.mode != 'RGB':
            im = im.convert('RGB')
        im.save(output_image)


if __name__ == '__main__':
    def parse_args():
        parser = ArgumentParser()
        parser.add_argument('-p', '--path_to_files', type=str, required=False, default=None,
                            help='Path to the folder with files')
        parser.add_argument('-f', '--input_files', type=list, required=False, default=None,
                            help='List of paths of the files')
        parser.add_argument('-r', '--reference', type=str, required=False, default=None,
                            help='Path of the reference file')
        parser.add_argument('-rn', '--reference_name', type=str, required=False, default=None,
                            help='Name of reference file')
        parser.add_argument('-o', '--output_path', type=str, required=False, default='df_data',
                            help='Path to the output folder')
        parser.add_argument('-m', '--matching', type=bool, required=False, default=False,
                            help='Method of matching all files combined')
        parser.add_argument('-ms', '--matching_single', type=bool, required=False, default=False,
                            help='Method of matching a reference to the rest of files')
        parser.add_argument('-d', '--debug', type=bool, required=False, default=False,
                            help='Debug mode')
        return parser.parse_args()


    args = parse_args()

    dF = duplicateFinder(debug=args.debug, output_path=args.output_path)
    dF.run_on_files(path_to_files=args.path_to_files,
                    input_files=args.input_files,
                    reference=args.reference,
                    reference_name=args.reference_name,
                    matching=args.matching,
                    matching_single=args.matching_single)

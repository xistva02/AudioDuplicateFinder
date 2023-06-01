Audio Duplicate Finder

Introduction:
This repository contains the "duplicateFinder", a script for finding audio duplicates inside given set of recordings.
Unlike many other approaches, it uses raw audio information without any metadata (except name of files) and is able to recognise
duplicates even when they slightly differ and/or contain only a segment of a recording.

In this repo, you will find two main scripts: df.py and df_GUI.py---both contains the same functionality, either called by
flags or a simple tkinter GUI.

Description:
The pipeline of the duplicateFinder can be divided into few steps. First, chroma representation of all files is computed
via either synctoolbox or nnAudio modules and saved in ./df_data/chroma folder (if matching_single is set to True, its chroma vectors and images are saved in ./df_data/reference folder.
If nnAudio is selected (default) and cuda driver on your device is available, the computation of chroma vectors is much faster. Then, all chroma files are converted to the .png images and saved in ./df_data/img folder.
All selected images (or a single image, depending on the preset) are compared with all images. This is done by opencv matchTemplate and minMaxLoc methods.
It works also for a small segment of given audio file---e.g., first 10 seconds of a recording compared to the whole recording is a match (duplicate) and should be evaluated this way.
Finally, results are printed to the console (or GUI console) and saved in ./df_data/output folder. The name of output .csv and .xlsx file depends on the method used (such as df_matching_hard_duplicates.csv).

The program computes 'hard' duplicates, 'soft' duplicates and 'low' duplicates. It is possible that two recordings are similar in structure but not exactly the same -- parameter soft_val (default is set to 0.5) controls the threshold value.
Low duplicates show files that may be a bit similar. All duplicate types are handled and exported separately.
The implementation was tested on Windows 10 (21H2) and Linux Mint 21.1.

Matching and matching_single cannot be set to True in the same time. Similarly, reference and reference_name cannot be set in the same time. However, one can choose path_to_files and input_files simultaneously, automatically discarding the possible duplicate paths.

To install the dependencies to your virtual environment, use:
python install -r requirements.txt

Example of usage:
Let's have all audio files in /home/finder_data folder (and subfolders). To find all duplicates and use all vs. all strategy, use:
python dF.py -p /home/finder_data -m True

or if we want to specify output folder, use -o flag:

python dF.py -p /home/dF/finder_data -m True -o /home/dF/outputs

To select a specific file (e.g., test_recording.wav) and compare it to all other files in /home/df/finder_data folder, use:
python dF.py -p /home/finder_data -r /home/df/finder_data/test_recording.wav -ms True
or
python dF.py -p /home/df/finder_data -rn test_recording -ms True

df_GUI.py version works the same way, parameters are chosen via graphical interface.

Note that the code is not optimized or clean. It is a result of experimentation. The feedback and comments are appreciated! If you have any questions, feel free to open issue or send message to matej.istvanek@vut.cz.

This work was supported by the project TA ČR, TL05000527 Memories of Sound: The Evolution of the Interpretative Tradition
Based on the Works of Antonin Dvorak and Bedrich Smetana and was co financed with state support of the Technology Agency
of the Czech Republic within the ÉTA Program.
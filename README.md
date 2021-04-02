# DOCUMENT READER - IMAGE To TEXT
Document Reader is a well known problem in the field of computer vision and pattern recognition, where the objective is to convert the images with english text to editable text files. We are required to extract all the text from the image so that it can be used for searching purposes and also create an overlay so that text can be copied and pasted from the images saved as editable text documents.

This project is developed as a part of the "CSL7360 Computer Vision" course at IIT Jodhpur.

## Files
- Sub-directory training/ : contains the python file for training the model
- Sub-directory model/ : contains the trained model
- Sub-directory SampleImages/ : few samples from the dataset
- Sub-directory include/ & src/ : contains the cpp code for the application
- Poster.pdf : The project idea, presented in the form of a poster.
- Report.pdf : The short project report explaining the woking and functionalities.

## Compilation and Running
run the command: 
```
. ./make.sh
```
A binary executable file will be created in sub-directory bin/

To Run : 
```
./bin/binary <image-name> <model-file> <output-path>
```

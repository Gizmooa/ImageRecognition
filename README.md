# ImageRecognition

1) Create neg.txt
Start by running the script "cascadeHelper.py" to generate a neg.txt containing the filepaths of all
negative pictures in negative/

2) Create pos.txt
Download a openCV version with Haar, any version 3.4.XX should do. We want to use the 
program "opencv_annotation.exe". Stand inside the directory where you have the folder containing the positive
images and run [...]/opencv/build/x64/vc15/bin/opencv_annotation.exe --annotations=pos.txt --images=positives/
Where [...] is everything infront of the path, and positives is the name of the folder containing the positive images.
Now we have to locate every positive entry by drawing a rectangle. This will create the pos.txt. (Remember to change \ to / in pos.txt)

3) Create pos.vec
Here we want to use the "opencv_createsamples.exe" program. w, h, and num are all variables to play around with. 
by giving num a high value it will just take all the samples, but you can also make less. Run the following:
[...]opencv/build/x64/vc15/bin/opencv_createsamples.exe -info pos.txt -w 20 -h 20 -num 1000 -vec pos.vec

4) Train our cascade classifier
Here we want to use the "opencv_traincascade.exe" program. An example of training: (Results will be dropped into the parameter -data, and h, w has to be the same as above)
[...]/opencv/build/x64/vc15/bin/opencv_traincascade.exe -data cascadeTraining/ -vec pos.vec -bg neg.txt -numPos 200 -numNeg 100 -numStages 10 -w 20 -h 20
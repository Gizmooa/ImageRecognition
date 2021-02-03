# ImageRecognition in OSRS

### About the project
First of all, this is a beginner project and the first time I have worked with computer vision. The foundation of the project is based on guides from "Learn Code By Gaming" on YouTube, and has been very helpful. In this project you'll find two folders; HSVDetection and cascadeHSVAgent. 

##### HSVDetection
HSVDetection is purely meant for detection. Given a window name, needle image and a HSVFilter, it will draw rectangles on top of every object with a confidentiality of more than 50%. (In this case it will draw rectangles on every Copper Ore in OSRS)

##### cascadeHSVAgent
This program will use multiple positive and negative sample images, to build a cascade classifier model to detect iron ores and automatically mine the nearest one. In the last part of this readme I'll provide how to train more data, as the amount of data for this project is lacking quite a bit. 
For this problem I used both HSV filtering and cascade classifier together, but this is not necessary if one were to make more and better samples. But it was also an experiment to see if the amount of frames would decrease drastically. In the future I will maybe implement a dropping feature. This can be done by having a needle image of a full inventory, and look if that image is in the screenshot. (Or looking for the "Not enough space in inventory" tooltip)

##### How to train more data
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

To generate positive and negative images, you can simply set "GET_DATA" to true in cascadeHSVAgent/main.py. This allows you to press 'p' and 'f' to add positive and negative images to the folders cascadeHSVAgent/positives/ and cascadeHSVAgent/negatives/

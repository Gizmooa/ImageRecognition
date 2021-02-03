# Using Image Recognition to bot in OSRS

### About the project
First of all, this is a beginner project and the first time I have worked with computer vision. The foundation of the project is based on guides from "Learn Code By Gaming" on YouTube, and has been very helpful. In this project you'll find two folders; HSVDetection and cascadeHSVAgent. 

##### HSVDetection
This program will do image processing and object detection in real time on BlueStacks running OSRS. (This can easily be changed to other games, or applications) The image processing is done with a HSV filter, where we will use OpenCV to match a needle image on top of every screenshot. The program will then draw rectangles on every object with a confidentiality score higher than 50%. (This program will detect Copper Ores in OSRS)

##### cascadeHSVAgent
This program is a mining agent / bot for OSRS to power mine Iron Ores. It has 4 different states; INITIALIZING, SEARCHING, MINING, and DROPPING. The agent will find the closest iron ore deposite using pythagorean distance, mine it, and keep mining until a full inventory is reached. When a full inventory is reached, the agent will drop the whole inventory and switch to searching state and start all over. 

For this project I've used both HSV filtering and cascade classifier together, but this is far from necessary as I could've setted for one of them. But this was mostly to experiment with it. Training a good cascade classifier would've needed a lot more data than I've gathered in this project, which also is why this agent is acting extremely poorly in new locations. But it is working quite well in SE of Varrock. 

Notice that this agent / bot is no way near perfect. This project has been done for educational purposes only, and will most likely get your account banned if you use it. 

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

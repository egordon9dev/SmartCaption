# SmartCaptions
Captioning videos is no easy task. Automatic video captioning technology does not allow for intelligent placement of captions, and manual captioning requires much human effort for a simple task, and if captions are to be placed in such a way so as to not block objects of importance, then the process becomes even more tedious. In light of this problem, we have made SmartCaptions, a tool that intelligently places captions in relation to the entity being tracked, while not obscuring the entity from view. SmartCaptions uses face detection, facial recognition, object tracking, and caption placement. With a single object, proper camera angles, and good lighting, we are able to consistently place captions near a relevant object and style captions based on which character is recognized.

Contributers: Ethan Gordon, Jason Lee, Alex Liu, Aaditya Raghavan, Andrew Zhao

[project video]: https://youtu.be/yWbuJBaxGW4

## Troubleshooting

Get Frames:

This step requires the ffmpeg command. If it's not installed, please do so first.

To get the frames for a video of your choice, do the following:

- `sudo chmod +x ./store_frames.sh` (to allow permission to run the script)
- `./store_frames.sh path/to/video` (where the "path to video" is replaced by the absolute or relative path to the video file to process)

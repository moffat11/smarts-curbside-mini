"# Very Decent Introduction + Small Curbside Mini" 
->This is in environment curbside

->Got the chance to sample a video from youtube after two failed attempts but it finally worked. Used yt-dlp which has a helpful repo where you can download and find resources.
("https://github.com/yt-dlp/yt-dlp")

->Aftwards implemented four programs programs: detect.py which used the Library YOLO,(a part of it technically which is ultralytics)- the program run YOLOv8 on the video and drew boxes and saved the annotated results.[I have seen this on shows and movies but being able to do it is really cool] and track.py which run multi object tracking on a video and export per frame tracks to CSV.

->The third program is summarize_tracks.py which
    -reads tracks.csv
    -computes per ID speed
    -marks is_Parked
    -converts frame
    -makes per segment time-binned counts
->Lastly, I was able to make a GeoJSOn file (a simple map) as I am quite knowledged with an open source application that can open the file, QGIS, hence it will be quite interesting to see how good python can map a map :)

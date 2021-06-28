<h1 align="center">
<br>
<a href="https://github.com/shivang02/FaceMouse"><img src="./misc_assets/FaceMouseLogoRound.png" alt="Face Mouse Logo" width="200"></a>
<br>
  Face Mouse
  <br>
</h1>

<h4 align="center">A virtual mouse pointer capable of moving cursor and performing clicks using facial landmarks detected in video stream</h4>

<p align="center">
  <a href="#key-features">Key Features</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#download">Download</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>



## Key Features

- **Control your cursor by moving your face**-
the predictor tracks the tip of your nose and moves the cursor in the direction the nose moves.
- **Perform click functions by blinking**-
	- Single blink (lasting 1-5 frames @ ~100-200fps video-stream) – *left click*
	- Double blink (lasting 1-5 frames @ ~100-200fps video-stream) – *double left click*
	- Triple blink (lasting 1-5 frames @ ~100-200fps video-stream) –*press mouse down* (for scroll and drag)
	- (any) Blink (after Triple blink) (lasting 1-5 frames @ ~100-200fps video-stream) – *release mouse down*
	- (long) Single blink (lasting >5 frames @ ~100-200fps video-stream) – *right click*


## How To Use

To clone and run this application, you'll need [Python 3](https://www.python.org/) installed on your computer.
From your command line:

```bash
# Clone this repository
$ git clone https://github.com/shivang02/FaceMouse.git

# Go into the repository
$ cd FaceMouse

# Install dependencies
$ pip install -r requirements.txt

# Run the app
$ python3 face_mouse.py 
```

## Download

You can [download](https://github.com/shivang02/FaceMouse/releases/tag/v1.0) the latest installable version of [FaceMouse](https://github.com/shivang02/FaceMouse) for Windows.

## Credits

This software mainly uses the following open source packages (among others):

- [OpenCV](https://opencv.org/)
- [Dlib](http://dlib.net/)
- [NumPy](https://numpy.org/)
- [PyAutoGUI](https://pypi.org/project/PyAutoGUI/)
- [Mouse](https://pypi.org/project/mouse/)

The following sources helped with some logic in the code:

- [Eye blink detection with OpenCV, Python, and dlib](https://www.pyimagesearch.com/2017/04/24/eye-blink-detection-opencv-python-dlib/)

## License

MIT

---

> GitHub &nbsp;&middot;&nbsp; [@shivang02](https://github.com/shivang02)
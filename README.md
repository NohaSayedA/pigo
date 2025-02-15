<h1 align="center"><img alt="pigo-logo" src="https://user-images.githubusercontent.com/883386/55795932-8787cf00-5ad1-11e9-8c3e-8211ba9427d8.png" height=240/></h1>

[![Build Status](https://travis-ci.org/esimov/pigo.svg?branch=master)](https://travis-ci.org/esimov/pigo)
[![GoDoc](https://godoc.org/github.com/golang/gddo?status.svg)](https://godoc.org/github.com/esimov/pigo/core)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?style=flat)](./LICENSE)
[![release](https://img.shields.io/badge/release-v1.1.0-blue.svg)](https://github.com/esimov/pigo/releases/tag/v1.1.0)
[![snapcraft](https://img.shields.io/badge/snapcraft-v1.1.0-green.svg)](https://snapcraft.io/pigo)
[![Snap Status](https://build.snapcraft.io/badge/esimov/pigo.svg)](https://build.snapcraft.io/user/esimov/pigo)

Pigo is a purely Go face detection library based on ***Pixel Intensity Comparison-based Object detection*** paper (https://arxiv.org/pdf/1305.4537.pdf). 

| Rectangle face marker | Circle face marker
|:--:|:--:
| ![rectangle](https://user-images.githubusercontent.com/883386/40916662-2fbbae1a-6809-11e8-8afd-d4ed40c7d4e9.png) | ![circle](https://user-images.githubusercontent.com/883386/40916683-447088a8-6809-11e8-942f-3112c10bede3.png) |

### Motivation
I've intended to implement this face detection method, since the only existing solution for face detection in the Go ecosystem is using bindings to OpenCV, but installing OpenCV on various platforms is sometimes daunting. 

This library does not require any third party modules to be installed. However in case you wish to try the real time, webcam based face detection you might need to have Python2 and OpenCV installed, but **the core API does not require any third party module or external dependency**. 

### Key features
- [x] Does not require OpenCV or any 3rd party modules to be installed
- [x] High processing speed
- [x] There is no need for image preprocessing prior detection
- [x] There is no need for the computation of integral images, image pyramid, HOG pyramid or any other similar data structure
- [x] The face detection is based on pixel intensity comparison encoded in the binary file tree structure
- [x] **Fast detection of in-plane rotated faces**

**The API can detect even faces with eyeglasses.**

![output](https://user-images.githubusercontent.com/883386/44484795-67e18a80-a657-11e8-98a1-06811dd7015c.png)

**The API can also detect in plane rotated faces.** For this reason a new `-angle` parameter have been included into the command line utility. The command below will generate the following result (see the table below for all the supported options).

```bash
$ pigo -in input.jpg -out output.jpg -cf data/facefinder -angle=0.8 -iou=0.01
```

| Input file | Output file
|:--:|:--:
| ![input](https://user-images.githubusercontent.com/883386/50761018-015db180-1272-11e9-93d9-d3693cae9d66.jpg) | ![output](https://user-images.githubusercontent.com/883386/50761024-03277500-1272-11e9-9c20-2568b87a2344.png) |


In case of in plane rotated faces the angle value should be adapted to the provided image.

## Install
Install Go, set your `GOPATH`, and make sure `$GOPATH/bin` is on your `PATH`.

```bash
$ export GOPATH="$HOME/go"
$ export PATH="$PATH:$GOPATH/bin"
```
Next download the project and build the binary file.

```bash
$ go get -u -f github.com/esimov/pigo/cmd/pigo
$ go install
```
### Binary releases
Also you can obtain the generated binary files in the [releases](https://github.com/esimov/pigo/releases) folder in case you do not have installed or do not want to install Go.

The library can be accessed as a snapcraft function too.

<a href="https://snapcraft.io/pigo"><img src="https://raw.githubusercontent.com/snapcore/snap-store-badges/master/EN/%5BEN%5D-snap-store-white-uneditable.png" alt="snapcraft caire"></a>

## API
Below is a minimal example of using the face detection API. 

First you need to load and parse the binary classifier, then convert the image to grayscale mode, 
and finally to run the cascade function which returns a slice containing the row, column, scale and the detection score.

```Go
cascadeFile, err := ioutil.ReadFile("/path/to/cascade/file")
if err != nil {
	log.Fatalf("Error reading the cascade file: %v", err)
}

src, err := pigo.GetImage("/path/to/image")
if err != nil {
	log.Fatalf("Cannot open the image file: %v", err)
}

pixels := pigo.RgbToGrayscale(src)
cols, rows := src.Bounds().Max.X, src.Bounds().Max.Y

cParams := pigo.CascadeParams{
	MinSize:     fd.minSize,
	MaxSize:     fd.maxSize,
	ShiftFactor: fd.shiftFactor,
	ScaleFactor: fd.scaleFactor,
	
	ImageParams: pigo.ImageParams{
		Pixels: pixels,
		Rows:   rows,
		Cols:   cols,
		Dim:    cols,
	},
}

pigo := pigo.NewPigo()
// Unpack the binary file. This will return the number of cascade trees,
// the tree depth, the threshold and the prediction from tree's leaf nodes.
classifier, err := pigo.Unpack(cascadeFile)
if err != nil {
	log.Fatalf("Error reading the cascade file: %s", err)
}

angle := 0.0 // cascade rotation angle. 0.0 is 0 radians and 1.0 is 2*pi radians

// Run the classifier over the obtained leaf nodes and return the detection results.
// The result contains quadruplets representing the row, column, scale and detection score.
dets := classifier.RunCascade(cParams, angle)

// Calculate the intersection over union (IoU) of two clusters.
dets = classifier.ClusterDetections(dets, 0.2)
```

## Usage
A command line utility is bundled into the library to detect faces in static images.

```bash
$ pigo -in input.jpg -out out.jpg -cf data/facefinder
```

### Supported flags:

```bash
$ pigo --help
┌─┐┬┌─┐┌─┐
├─┘││ ┬│ │
┴  ┴└─┘└─┘

Go (Golang) Face detection library.
    Version: 1.1.0

  -angle float
    	0.0 is 0 radians and 1.0 is 2*pi radians
  -cf string
    	Cascade binary file
  -circle
    	Use circle as detection marker
  -in string
    	Source image
  -iou float
    	Intersection over union (IoU) threshold (default 0.2)
  -json
    	Output face box coordinates into a json file
  -max int
    	Maximum size of face (default 1000)
  -min int
    	Minimum size of face (default 20)
  -out string
    	Destination image
  -scale float
    	Scale detection window by percentage (default 1.1)
  -shift float
    	Shift detection window by percentage (default 0.1)

```

### Real time face detection

In case you wish to test the library real time face detection capabilities using a webcam, the `examples` folder contains a  Web and a few Python examples. Prior running it you need to have Python2 and OpenCV2 installed.

To run the Python version:
```bash
$ python2 demo.py
```

To run the web version:

```bash
$ go run main.go -cf "../../data/facefinder"
```

Then access the `http://localhost:8081/cam` url from a web browser.


## Author

* Endre Simo ([@simo_endre](https://twitter.com/simo_endre))

## License

Copyright © 2018 Endre Simo

This software is distributed under the MIT license. See the LICENSE file for the full license text.

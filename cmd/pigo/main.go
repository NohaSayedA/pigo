package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"image"

	"github.com/NohaSayedA/pigo/core"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"time"
)

const banner = `
┌─┐┬┌─┐┌─┐
├─┘││ ┬│ │
┴  ┴└─┘└─┘

Go (Golang) Face detection library.
    Version: %s

`

// Version indicates the current build version.
var Version string

// detectionResult contains the coordinates of the detected faces and the base64 converted image.
type detectionResult struct {
	coords []image.Rectangle
}

func main() {
	var (
		// Flags
		source       = flag.String("in", "", "Source image")
		destination  = flag.String("out", "", "Destination image")
		cascadeFile  = flag.String("cf", "", "Cascade binary file")
		minSize      = flag.Int("min", 20, "Minimum size of face")
		maxSize      = flag.Int("max", 1000, "Maximum size of face")
		shiftFactor  = flag.Float64("shift", 0.1, "Shift detection window by percentage")
		scaleFactor  = flag.Float64("scale", 1.1, "Scale detection window by percentage")
		angle        = flag.Float64("angle", 0.0, "0.0 is 0 radians and 1.0 is 2*pi radians")
		iouThreshold = flag.Float64("iou", 0.2, "Intersection over union (IoU) threshold")
		circleMarker = flag.Bool("circle", false, "Use circle as detection marker")
		outputAsJSON = flag.Bool("json", false, "Output face box coordinates into a json file")
	)
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, fmt.Sprintf(banner, Version))
		flag.PrintDefaults()
	}
	flag.Parse()

	if len(*source) == 0 || len(*destination) == 0 || len(*cascadeFile) == 0 {
		log.Fatal("Usage: pigo -in input.jpg -out out.png -cf data/facefinder")
	}

	fileTypes := []string{".jpg", ".jpeg", ".png"}
	ext := filepath.Ext(*destination)

	if !inSlice(ext, fileTypes) {
		log.Fatalf("Output file type not supported: %v", ext)
	}

	if *scaleFactor < 1.05 {
		log.Fatal("Scale factor must be greater than 1.05")
	}

	// Progress indicator
	s := new(spinner)
	s.start("Processing...")
	start := time.Now()

	fd := pigo.NewFaceDetector(*destination, *cascadeFile, *minSize, *maxSize, *shiftFactor, *scaleFactor, *iouThreshold, *angle)
	faces, err := fd.DetectFaces(*source)
	if err != nil {
		log.Fatalf("Detection error: %v", err)
	}

	_, rects, err := fd.DrawFaces(faces, *circleMarker)

	if err != nil {
		log.Fatalf("Error creating the image output: %s", err)
	}

	resp := detectionResult{
		coords: rects,
	}

	out, err := json.Marshal(resp.coords)

	if *outputAsJSON {
		ioutil.WriteFile("output.json", out, 0644)
	}

	s.stop()
	fmt.Printf("\nDone in: \x1b[92m%.2fs\n", time.Since(start).Seconds())
}

type spinner struct {
	stopChan chan struct{}
}

// Start process
func (s *spinner) start(message string) {
	s.stopChan = make(chan struct{}, 1)

	go func() {
		for {
			for _, r := range `-\|/` {
				select {
				case <-s.stopChan:
					return
				default:
					fmt.Printf("\r%s%s %c%s", message, "\x1b[92m", r, "\x1b[39m")
					time.Sleep(time.Millisecond * 100)
				}
			}
		}
	}()
}

// End process
func (s *spinner) stop() {
	s.stopChan <- struct{}{}
}

// inSlice check if a slice contains the string value.
func inSlice(ext string, types []string) bool {
	for _, t := range types {
		if t == ext {
			return true
		}
	}
	return false
}

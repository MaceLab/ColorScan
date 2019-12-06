'''
ColorScan
By Ray Parker
Mace Lab, Tufts University
November 26 2019

This software is designed for colorimetric analysis of paper-based microfluidic
devices. It is written for Python 3, with the libraries NumPy, Matplotlib,
TkInter, PIL (Pillow), and OpenCV. All packages other than OpenCV are included
in the Anaconda distribution of Python, which we recommend. OpenCV is available
for installation through Anaconda's package manager, Conda.

'''

print("Loading packages, please wait...")


import numpy as np #for array operations
import cv2 #for image processing
import tkinter as tk #for GUI
import tkinter.filedialog #must be explicitly imported
from tkinter import ttk #for an updated visual style
from PIL import Image, ImageTk #for making openCV images able to be displayed by tkinter
import os #for filepath operations
import sys #for commandline arguments

import matplotlib #For plotting histograms
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt




#These are required to replace the missing constants in the OpenCV 3
CV_CONTOURS_MATCH_I1 = 1 #sum of absolute differences of reciprocals of each image invariant
CV_CONTOURS_MATCH_I2 = 2 #sum of absolute differences of each image invariant
CV_CONTOURS_MATCH_I3 = 3 #greatest absolute difference of image invariants, normalized to A


#Keycodes for keyboard events
ENTER = 13
ESC = 27
TAB = 9

sizeFudge = 30 #Number of pixels to decrease the size of the displayed image by to avoid overlapping borders


INVALID_PRESET_NUM = -999999 #A number that will (hopefully) never show up in a valid preset


RGB2grayscale_weights = np.array([0.299, 0.587, 0.114]) #RGB weights to convert to grayscale
#See https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html for reference

root = tk.Tk() #initializing the root window
try:
    #If the spotomatic icon is placed in the folder, uses it
    root.iconbitmap(default=r'ColorScanIcon.ico')
except tk.TclError:
    #If the icon is missing, doesn't use it
    pass


#Function: windowAspectAdjust
#Arguments: frame size, input image, scaling parameter to avoid filling the whole screen
#Purpose: to resize the image to fit the screen, while maintaining the same aspect ratio
#Returns: new width, height (ints) [pixels]
def windowAspectAdjust(frameSize, im, scaling=2/3):
    screen_width, screen_height = frameSize
    imHeight = im.shape[0]
    imWidth = im.shape[1]
    if imHeight/screen_height>imWidth/screen_width:
        windHeight = screen_height
        windWidth = screen_height*imWidth/imHeight
    else:
        windHeight = screen_width*imHeight/imWidth
        windWidth = screen_width
        
    return int(windWidth*scaling), int(windHeight*scaling)




#takes in a 2D array of integers (image channel) and a boolean mask of the same size
#returns a histogram of the values for that channel in the masked area
def imageChannelHistogram(channel, mask=None, bins=256):
    if mask is not None:
        channel_masked = np.ma.MaskedArray(channel, ~mask).compressed()
    else:
        channel_masked = channel
    heights, edges = np.histogram(channel_masked, bins, (0,256))
    return heights, edges


#Produces the vertices of a regular polygon, where x = cos(2 i (pi) k/n) and y = sin(2 i (pi) k/n) for
#  k from 0 to n-1 for open polygon (toClose=False) or from 0 to n for closed polygon (toClose=True)
#  argument phi rotates the polygon, argument center shifts the origin, r controls the radius of the circumscribing circle
def regularPolygonPoints(n, phi=0, center=np.array([0,0]), r=1, toClose=False):
    angles = np.array([2*np.pi*k/n+phi for k in range(n+int(toClose))])
    points = np.array([r*np.cos(angles)+center[0], r*np.sin(angles)+center[1]]).T
    return points


#Generalized OpenCV shape drawing function for various possible geometries
#Args: image array (numpy) im, string shape ['polygon','rectangle','circle']
#  numpy 2D vector for center coordinates of the shape, list of data for the shape:
#     data organkzation: [general shape arguments, [size arguments]]
#     Ex: polygon:   [# of sides, angle of polygon, [radius]]
#         circle:    [[radius]]
#         rectangle: [[width, height]]
def drawShape(im, shape, center, data, color, thickness):
    
    if shape=='polygon':

        n_sides = data[0]
        angle = data[1]
        radius = data[2][0]

        points = regularPolygonPoints(n_sides, angle, center, radius)

        
        if thickness>=0:
            cv2.polylines(im, np.array([points]).astype(np.int32), True, color, thickness)
        else:
            cv2.fillConvexPoly(im, np.array([points]).astype(np.int32), color)
                                        
    elif shape=='rectangle':

        width = data[0][0]
        height = data[0][1]

        #cv2 Rectangles are defined by two points (stored as tuples)
        tl = tuple((center-np.array([width/2,height/2])).astype(int)) 
        br = tuple((center+np.array([width/2,height/2])).astype(int))

        
        cv2.rectangle(im, tl, br, color, thickness)

    elif shape=='circle':

        radius = data[0][0]
        #cv2 Circles are defined by a center (tuple) and a radius
        cv2.circle(im, tuple(center.astype(int)), radius, color, thickness)
        
        
    else:
        raise NotImplementedError(f"Shape {shape} not implemented!")
            





#Object: AnalysisWindow
#Purpose: Analysis window object to contain tkinter objects and opencv analysis methods
class AnalysisWindow:

    def __init__(self, window, base):
        
        self.window = window
        self.base = base
        window.title("Analysis Menu")

        
        #Path to the presets file, stored as a binary file by numpy
        self.presetPath = 'presets.npy'
        
        #Settings that will be able to be preset
        #New settings must start with 'V_' to be recognized
        #All settings must be tkinter variables
        self.V_maskThresh1 = tk.IntVar(value=0)
        self.V_maskThresh2 = tk.IntVar(value=0)
        self.V_maskMode = tk.IntVar(value=0)
        self.V_dilerocode = tk.StringVar(value="")
        self.V_blurAmount = tk.IntVar(value=1)
        self.V_sizeTol = tk.DoubleVar(self.window, value=20)
        self.V_shapeTol = tk.DoubleVar(self.window, value=1)
        self.V_saveRGB = tk.BooleanVar(value=True)
        self.V_saveHSV = tk.BooleanVar(value=True)
        self.V_saveLAB = tk.BooleanVar(value=True)
        self.V_saveHistograms = tk.BooleanVar(value=False)
        self.V_refiner_displace_x = tk.IntVar(value=INVALID_PRESET_NUM)
        self.V_refiner_displace_y = tk.IntVar(value=INVALID_PRESET_NUM)
        self.V_refiner_radius = tk.IntVar(value=INVALID_PRESET_NUM)

        #Fills the array self.presetArray with preset data from the presets file
        self.getPresets()

        #Variable to store the name of the current preset
        self.currentPreset = tk.StringVar(master=self.window)

        #List of names of current presets
        self.presetNames = self.presetArray['PresetName'] #pulls out the list of available presets
        
        self.currentPreset.set("Default")

        

        ######Building the gui######


        #Variable to store what level of analyses to display
        #   0 -> base image
        #   1 -> mask image
        #   2 -> dilate/erode result
        #   3 -> blur result
        self.showWhat = tk.IntVar(self.window, value=0)


        #Row variable for more control of where tkinter grids things, allows for easy restructuring
        ###Top Row###
        row = 0


        #Toggle button for original image
        self.imageToggle = tk.Radiobutton(self.window, text = "Original Image", value=0,
                                          variable=self.showWhat, indicatoron=False, command=self.updateImage)
        self.imageToggle.grid(column=0, row=row, sticky='we')

        #Copies image over from base so we don't accidentally change anything
        #Allows for successive analyses of the same image without closing the program
        self.im = self.base.image.copy()

        #Coversions of the image into various colorspaces for easier analysis
        self.imHSV = cv2.cvtColor(self.im, cv2.COLOR_BGR2HSV)
        self.imLAB = cv2.cvtColor(self.im, cv2.COLOR_BGR2LAB)
        

        #The image that will be displayed at each step of analysis
        self.dispIm = self.im.copy()

        #Will be the sum of all the contour masks
        self.totalMask = np.zeros(self.im.shape[:2], dtype=np.uint8)


        #Setting up gui for preset selection
        self.presetLabel = ttk.Label(self.window, text="Preset:")
        self.presetLabel.grid(row=row, column=1, sticky='e')
        self.presetFrame = ttk.Frame(self.window)
        self.presetFrame.grid(row=row, column=2, sticky='we')


        style = ttk.Style()
        style.configure("TMenubutton", relief=tk.RAISED, borderwidth=3)

        self.presetMenu = ttk.OptionMenu(self.presetFrame, self.currentPreset, *self.presetNames, command=self.loadPreset)        
        self.presetMenu.grid(row=0, column=0, sticky='we')
        self.savePresetButton = ttk.Button(self.presetFrame, text="Save New Preset", command=self.getNewPresetName)
        self.savePresetButton.grid(row=0, column=1, sticky='we')

        self.presetFrame.grid_columnconfigure(0, weight=1)
        self.presetFrame.grid_columnconfigure(1, weight=1)
        self.presetFrame.grid_columnconfigure(2, weight=1)


        

        row += 1
        ###Next Row###

        #Toggle button for showing the mask
        self.maskToggle = tk.Radiobutton(self.window, text="Masked Image", value=1,
                                         variable=self.showWhat, indicatoron=False, command=self.updateAnalyses)
        self.maskToggle.grid(row=row, column=0, sticky='we')

        #Setting for threshold 1 of the mask: value (brightness)
        self.maskThresh1Slider = ttk.Scale(self.window, from_=0, to=255, orient=tk.HORIZONTAL, command=self.updateAnalyses, variable=self.V_maskThresh1)
        self.maskThresh1Slider.grid(row=row, column=1, columnspan=1, sticky='we')
        self.maskThresh1LabelIndicatorFrame = ttk.Frame(self.window)
        self.maskThresh1LabelIndicatorFrame.grid(row=row, column=2, sticky='w')
        self.maskThresh1Indicator = ttk.Entry(self.maskThresh1LabelIndicatorFrame, textvariable=self.V_maskThresh1, width=5)
        self.maskThresh1Indicator.grid(row=0, column=0, sticky='w')
        self.maskThresh1Indicator.bind("<Key>", lambda e: self.updateAnalyses() if e.keycode==TAB or e.keycode==ENTER else 0)
        self.maskThresh1Label = ttk.Label(self.maskThresh1LabelIndicatorFrame, text="Value Threshold")
        self.maskThresh1Label.grid(row=0, column=1, sticky='w')




        row += 1
        ###Next Row###


        #Setting for threshold 2 of the mask: saturation
        self.maskThresh2Slider = ttk.Scale(self.window, from_=0, to=255, orient=tk.HORIZONTAL, command=self.updateAnalyses, variable=self.V_maskThresh2)
        self.maskThresh2Slider.grid(row=row, column=1, columnspan=1, sticky='we')
        self.maskThresh2LabelIndicatorFrame = ttk.Frame(self.window)
        self.maskThresh2LabelIndicatorFrame.grid(row=row, column=2, sticky='w')
        self.maskThresh2Indicator = ttk.Entry(self.maskThresh2LabelIndicatorFrame, textvariable=self.V_maskThresh2, width=5)
        self.maskThresh2Indicator.grid(row=0, column=0, sticky='w')
        self.maskThresh2Indicator.bind("<Key>", lambda e: self.updateAnalyses() if e.keycode==TAB or e.keycode==ENTER else 0)
        self.maskThresh2Label = ttk.Label(self.maskThresh2LabelIndicatorFrame, text="Saturation Threshold")
        self.maskThresh2Label.grid(row=0, column=1, sticky='w')


        row += 1
        ###Next Row###

        #Toggle for whether to AND the masks generated by each threshold or OR them
        #(AND gets faster performance)
        self.maskModeSelectorLabel = ttk.Label(self.window, text="Threshold mode:")
        self.maskModeSelectorLabel.grid(row=row, column=0, sticky='we')
        self.maskModeSelectorAND = ttk.Radiobutton(self.window, text="AND", value=0, variable=self.V_maskMode, command=self.updateAnalyses)
        self.maskModeSelectorAND.grid(row=row, column=1)
        self.maskModeSelectorOR = ttk.Radiobutton(self.window, text="OR", value=1, variable=self.V_maskMode, command=self.updateAnalyses)
        self.maskModeSelectorOR.grid(row=row, column=2)


        row += 1
        ###Next Row###


        #Setup for the dilation/erosion step

        #Function: setDileroMode, dilates/erodes mask depending on how many times the user has pressed each button
        #args: dileromode: variable to store what type of transformation to apply at a dilation/erosion step
        def setDileroMode(dileromode):
            #Sets the analysis to show output of dialtion/erosion
            self.showWhat.set(2)

            #dilerocode stores the number and order of dilations/erosions (preset variable)
            if dileromode==1:
                self.V_dilerocode.set(self.V_dilerocode.get()+'d')
            elif dileromode==-1:
                self.V_dilerocode.set(self.V_dilerocode.get()+'e')
            elif dileromode==0:
                self.V_dilerocode.set("")

            #Updates analysis based on input
            self.updateAnalyses()

        #Dilation button
        self.dilateButton = ttk.Button(self.window, text = "Dilate", command = lambda: setDileroMode(1))
        self.dilateButton.grid(row=row, column=0, sticky='we')

        #Erosion button
        self.erodeButton = ttk.Button(self.window, text = "Erode", command = lambda: setDileroMode(-1))
        self.erodeButton.grid(row=row, column=1, sticky='we')

        #Reset button, clears all dilations/erosions
        self.resetButton = ttk.Button(self.window, text = "Reset", command = lambda: setDileroMode(0))
        self.resetButton.grid(row=row, column=2, sticky='we')

        row += 1
        ###Next Row###

        # Dilation counter, tells user how many times they have pressed the button
        #    (counter is updated when image is dilated/eroded)
        self.dilateCounter = tk.IntVar(value=0)        
        self.dilateIndicator = ttk.Label(self.window, textvariable=self.dilateCounter)
        self.dilateIndicator.grid(row=row, column=0)

        # Erosion counter, tells user how many times they have pressed the button
        #    (counter is updated when image is dilated/eroded)
        self.erodeCounter = tk.IntVar(value=0)
        self.erodeIndicator = ttk.Label(self.window, textvariable=self.erodeCounter)
        self.erodeIndicator.grid(row=row, column=1)



        row += 1
        ###Next Row###


        
        #Setup for blurring step
        self.blurToggle = tk.Radiobutton(self.window, text = "Blurred Mask", value=3,
                                         variable=self.showWhat, indicatoron=False, command=self.updateAnalyses)
        self.blurToggle.grid(row=row, column=0, sticky='we')

        #Setting for how much the mask should be blurred
        self.blurSlider = ttk.Scale(self.window, from_=1, to=10, orient=tk.HORIZONTAL, command=self.updateAnalyses, variable=self.V_blurAmount)
        self.blurSlider.grid(row=row, column=1, columnspan=1, sticky='we')
        self.blurIndicator = ttk.Entry(self.window, textvariable=self.V_blurAmount, width=5)
        self.blurIndicator.grid(row=row, column=2, sticky='w')
        self.blurIndicator.bind("<Key>", lambda e: self.updateAnalyses() if e.keycode==TAB or e.keycode==ENTER else 0)



        row += 1
        ###Next Row###


        #A blank line
        spacer = ttk.Label(master=self.window, text="")
        spacer.grid(row=row)

        #Spacer height will vary with size of window
        self.window.grid_rowconfigure(row, weight=1)


        row += 1
        ###Next Row###


        #Setup for the lists to store contours and contour properties
        self.contours = [] #contours
        self.sizes = [] #size of each contour
        self.closeInds = [] #indices of the contours which are similar to the selected one
        self.closeIndsPlus = [] #the second step of selection
        self.addConts = [] #manually added contours
        self.removeConts = [] #manually removed contours
        self.centers = [] #centroid of each contour

        #Setup for the lists to store information for individual contours post-refinement
        self.refinedMasks = [] #masks for each refined zone
        self.refinedCenters = [] #center of each refined zone

        #Setup for printing numbers on the screen after analysis
        self.numberTextArgs = [] #arguments for each cv2.putText call

        #Button to find the contours
        self.contourButton = ttk.Button(self.window, text = "Find Contours", command=self.cvContour)
        self.contourButton.grid(row=row, column=0, columnspan=2, sticky='we')

        #Checkbutton to ask whether to draw the contours on the screen or not
        self.drawConts = tk.BooleanVar(value=False)
        self.showContCheck = ttk.Checkbutton(self.window, text="Draw Contours", variable=self.drawConts, onvalue=True, command=self.updateAnalyses)
        self.showContCheck.grid(row=row, column=2, sticky='w')

        #These elements are disabled until there is a mask displayed
        self.contourButton.state(['disabled'])
        self.showContCheck.state(['disabled'])


        row += 1
        ###Next Row###


        #A counter for how many contours are found
        self.contourCount = ttk.Label(self.window, text = "")
        self.contourCount.grid(row=row, column=0, columnspan=3, sticky='w')

        #An indicator for the currently selected contour
        self.selectedContLabel = ttk.Label(self.window, text = "")
        self.selectedContLabel.grid(row=row, column=2, sticky='e')


        row += 1
        ###Next Row###

        #Button to find similar contours to the selected one
        self.similarContsButton = ttk.Button(self.window, text = "Find Similar Contours", command=self.getSimilarContours)
        self.similarContsButton.grid(row=row, column=0, columnspan=3, sticky='we')

        #Will be disabled until a contour is selected
        self.similarContsButton.state(['disabled'])


        row += 1
        ###Next Row###


        #Settings for the size tolerance threshold for ruling out dissimilar contours
        self.sizeTolLabel = ttk.Label(self.window, text = "Size Tolerance:")
        self.sizeTolLabel.grid(row=row, column=0, sticky='e')
        self.sizeTolSlider = ttk.Scale(master=self.window, from_=0, to=100,
                                       orient=tk.HORIZONTAL, command=self.getSimilarContours, variable=self.V_sizeTol)
        self.sizeTolSlider.grid(row=row, column=1, sticky='we')
        self.sizeTolIndicatorFrame = ttk.Frame(self.window)
        self.sizeTolIndicatorFrame.grid(row=row, column=2, sticky='we')
        self.sizeTolIndicator = ttk.Entry(self.sizeTolIndicatorFrame, textvariable=self.V_sizeTol, width=5)
        self.sizeTolIndicator.grid(row=0, column=0, sticky='we')
        percentLabel = ttk.Label(self.sizeTolIndicatorFrame, text='%')
        percentLabel.grid(row=0, column=1, sticky='w')

        self.sizeTolIndicator.bind("<Key>", lambda e: self.getSimilarContours() if e.keycode==TAB or e.keycode==ENTER else 0)

        #Will be disabled until the Find Similar Contours button is pressed
        self.sizeTolLabel.state(["disabled"])
        self.sizeTolSlider.state(["disabled"])
        self.sizeTolIndicator.state(["disabled"])


        row += 1
        ###Next Row###



        #Settings for the shape tolerance threshold for ruling out dissimilar contours
        self.shapeTolLabel = ttk.Label(self.window, text = "Shape Tolerance:")
        self.shapeTolLabel.grid(row=row, column=0, sticky='e')
        self.shapeTolSlider = ttk.Scale(master=self.window, from_=0, to=2,
                                        orient=tk.HORIZONTAL, command=self.getSimilarContours, variable=self.V_shapeTol)
        self.shapeTolSlider.grid(row=row, column=1, sticky='we')
        self.shapeTolIndicator = ttk.Entry(self.window, textvariable=self.V_shapeTol, width=5)
        self.shapeTolIndicator.grid(row=row, column=2, sticky='w')

        self.shapeTolIndicator.bind("<Key>", lambda e: self.getSimilarContours() if e.keycode==TAB or e.keycode==ENTER else 0)


        #Will be disabled until the Find Similar Contours button is pressed
        self.shapeTolLabel.state(["disabled"])
        self.shapeTolSlider.state(["disabled"])
        self.shapeTolIndicator.state(["disabled"])



        row += 1
        ###Next Row###


        spacer = ttk.Label(master=self.window, text="")
        spacer.grid(row=row)

        self.window.grid_rowconfigure(row, weight=1)
        

        row += 1
        ###Next Row###


        #Button to open the zone refinement dialog
        self.refineButton = ttk.Button(self.window, text='Refine Zones', command=self.refineZones)
        self.refineButton.grid(row=row, column=0, columnspan=2, sticky='we')
        self.refineButton.state(['disabled'])

        #Checkbutton to ask whether to show only the parts of the image that are not masked
        self.showRefinedZones = tk.BooleanVar(self.window, value=False)
        self.refineMaskShowCheck = ttk.Checkbutton(self.window, text='Mask Zones', variable=self.showRefinedZones, command=self.updateAnalyses)
        self.refineMaskShowCheck.grid(row=row, column=2, sticky='w')
        self.refineMaskShowCheck.state(['disabled'])
        

        row += 1
        ###Next Row###


        spacer = ttk.Label(master=self.window, text="")
        spacer.grid(row=row)

        self.window.grid_rowconfigure(row, weight=1)


        row += 1
        ###Next Row###
        

        #Big button to analyze the zones found
        self.analyzeButton = ttk.Button(master=self.window, text='Analyze',
                                        command=self.analyzeContours)
        self.analyzeButton.grid(row=row, column=0, rowspan=4, columnspan=2, sticky='nesw')

        #Will be disabled until image is ready to analyze (similar contours at minimum)
        self.analyzeButton.state(['disabled'])


        #Label for the list of various colorspace options
        self.outputLabel = ttk.Label(master=self.window, text="Output Colorspaces:")
        self.outputLabel.grid(row=row, column=2, sticky='e')
        self.outputLabel.state(['disabled'])



        row += 1
        ###Next Row###

        #Checkbutton to ask whether to output RGB
        self.RGBCheck = ttk.Checkbutton(master=self.window, text="RGB",
                                        variable=self.V_saveRGB, onvalue=True)
        self.RGBCheck.grid(row=row, column=2, sticky='e')
        self.RGBCheck.state(['disabled'])


        row += 1
        ###Next Row###

        #Checkbutton to ask whether to output HSV
        self.HSVCheck = ttk.Checkbutton(master=self.window, text="HSV",
                                        variable=self.V_saveHSV, onvalue=True)
        self.HSVCheck.grid(row=row, column=2, sticky='e')
        self.HSVCheck.state(['disabled'])


        row += 1
        ###Next Row###

        #Checkbutton to ask whether to output LAB
        self.LABCheck = ttk.Checkbutton(master=self.window, text="LAB",
                                        variable=self.V_saveLAB, onvalue=True)
        self.LABCheck.grid(row=row, column=2, sticky='e')
        self.LABCheck.state(['disabled'])


        row += 1
        ###Next Row###

        #Checkbutton to ask whether to output histograms
        self.histCheck = ttk.Checkbutton(master=self.window, text="Histograms",
                                        variable=self.V_saveHistograms, onvalue=True)
        self.histCheck.grid(row=row, column=2, sticky='e')
        self.histCheck.state(['disabled'])



        #Boolean to output the cropped image around each zone
        #   (currently no option to change this)
        self.saveCrops=True

        #Number of pixels to expand the border around the cropped zone
        self.saveBorder = 5

        #Variables to store the current position of the mouse on the screen
        self.mousex = None
        self.mousey = None

        #Variable to store the currently selected contour (by clicking)
        self.selectedCont = -1
        
        #Boolean to decide whether or not the mouse is in frame
        self.inFrame = tk.BooleanVar(value=False)
        self.base.display.bind("<Enter>", lambda e: self.inFrame.set(True))
        self.base.display.bind("<Leave>", lambda e: self.inFrame.set(False))


        #Making the columns all expand equally when the window is resized
        self.window.grid_columnconfigure(0, weight=1)
        self.window.grid_columnconfigure(1, weight=1)
        self.window.grid_columnconfigure(2, weight=1)

        #Setting the default display mode
        self.imageToggle.invoke()

        #Setting the default mask mode
        self.maskModeSelectorAND.invoke()



    #Clean up the window and then close it
    def close(self):
        self.base.display.unbind("<Motion>")
        self.base.display.unbind("<Button-1>")

        self.window.destroy()


    #Stores the array of presets to the presetArray variable,
    #   or makes a new file to store presets in
    def getPresets(self):
        try:
            self.presetArray = np.load(self.presetPath)

        #If it can't find the saved presets it makes a new file for presets, including Default
        except FileNotFoundError:
            self.presetArray = []
            self.savePreset('Default')


    #Opens a dialog box to get the user input name of a new preset
    def getNewPresetName(self):

        #####Building the dialog box GUI#####
        
        dialogBox = tk.Toplevel(master=self.window)
        dialogBox.title("")
        dialogText = ttk.Label(master=dialogBox, text="Please enter name for new preset", justify=tk.CENTER)
        dialogText.grid(column=0, row=0, sticky='we')
        newPresetName = tk.StringVar(value=self.currentPreset.get())

        #Defining a function to make sure it will fit in the box
        checkWidth = lambda s: len(s)<=16
        widthCheckCommand = self.window.register(checkWidth)

        
        entrybox = ttk.Entry(master=dialogBox, textvariable=newPresetName, width=17, validate='key', validatecommand=(widthCheckCommand, '%P'))
        entrybox.grid(column=0, row=1, sticky='we')
        buttonFrame = tk.Frame(master=dialogBox)
        buttonFrame.grid(column=0, row=2, sticky='we')
        doneButton = ttk.Button(master=buttonFrame, text="Done", command = lambda: [dialogBox.destroy(), self.savePreset(newPresetName.get())])
        doneButton.grid(row=0, column=0, sticky='we')
        cancelButton = ttk.Button(master=buttonFrame, text="Cancel", command = dialogBox.destroy)
        cancelButton.grid(row=0, column=1, sticky='we')

        entrybox.bind("<Key>", lambda e: [dialogBox.destroy(), self.savePreset(newPresetName.get())] if e.keycode==TAB or e.keycode==ENTER \
                      else dialogBox.destroy() if e.keycode==ESC else 0)


        dialogBox.grid_columnconfigure(0, weight=1)
        dialogBox.grid_rowconfigure(0, weight=1)
        buttonFrame.grid_columnconfigure(0, weight=1)
        buttonFrame.grid_columnconfigure(1, weight=1)


    #Stores the current preset in the presets file
    def savePreset(self, presetName):

        #list of names of the variables to be stored in the preset,
        #   starts with the field for name of the preset itself
        names = ['PresetName']
        #list of the data to be stored in each field
        values = [presetName]
        #list of the data types of each variable to be stored
        types = ['U16']

        #dir() returns the names of all the members of a class
        members = dir(self)
        for i in range(len(members)):
            #A nasty, nasty workaround to determine easily which variables to save
            #all saved variables must start with "V_"
            if members[i].startswith('V_'): 
                member = members[i]
                names.append(member)
                #All saved variables must have a .get() method
                #   (should all be tkinter variables)
                value = getattr(self, member).get()
                values.append(value)
                thetype = type(value)
                #numpy.save had trouble saving and loading things with type str
                if thetype==str:
                    thetype='U32'
                types.append(thetype)

        #Makes a new row for the preset array (a numpy structured array)
        presetArray_row = np.array([tuple(values)], dtype = {'names':names, 'formats':types})

        #If the preset array doesn't exist yet, make it
        if len(self.presetArray)==0:
            self.presetArray = presetArray_row

        #If the preset array already exists, append a row
        else:
            #If the preset to be saved doesn't already have a row, make it
            if presetName not in self.presetArray['PresetName']:
                self.presetArray = np.append(self.presetArray, presetArray_row)
                self.presetMenu.set_menu(presetName, *self.presetArray['PresetName'])
            #If the preset is already stored, update it
            else:
                self.presetArray[np.where(self.presetArray['PresetName']==presetName)[0]] = presetArray_row
        #Save the new preset array, don't pickle because of security
        np.save(self.presetPath, self.presetArray, allow_pickle=False)
        print(f'Saved preset {presetName}')




    #Opens a row in the preset array and stores its values to the corresponding variables
    def loadPreset(self, var):

        #The preset to be loaded, in the row corresponding to the selected preset name
        newPreset = self.presetArray[np.where(self.presetArray['PresetName']==self.currentPreset.get())[0]][0]

        #List of names of the variables in the preset
        names = newPreset.dtype.names

        for i in range(len(names)):
            name = names[i]
            val = newPreset[i]

            #Applying the nasty, nasty workaround
            if name.startswith("V_"):
                try:
                    #Preset variables must have a .set() method
                    #   (should all be tkinter variables)
                    getattr(self, name).set(str(val))
                
                #Sometimes the preset list will change between versions, this catches the errors
                except AttributeError:
                    print(f'preset {name} not found in current version, ignoring')

        #Update the analysis based on the new preset values
        self.updateAnalyses()


    #Updates the analysis of the image, depending on the selected mode
    def updateAnalyses(self, e=None):
        mode = self.showWhat.get()

        #if the original image is selected, no analysis steps will be applied
        self.analyzed = self.im.copy()

        #Each analysis step modifies self.analyzed in order

        #if the mask is selected
        if mode>0:
            self.cvMask()
        #if a dilation/erosion has been applied
        if mode>1:
            self.cvDilateErode()
        #if the blurred image is selected
        if mode>2:
            self.cvBlur()

        #Displays the analyzed image
        self.updateImage()

                    

    #Displays the analyses performed
    def updateImage(self, e=None):

        #If at least one analysis has been performed, must be converted to color from grayscale
        if self.showWhat.get()>0:        
            self.dispIm[:,:,:] = cv2.cvtColor(self.analyzed, cv2.COLOR_GRAY2BGR)
            self.contourButton.state(['!disabled'])
        #If no analyses have been performed the image is already color
        else:
            self.dispIm = self.im.copy()
            self.contourButton.state(['disabled'])

        #Will show only the zones that are included in the mask
        if self.showRefinedZones.get():
            self.dispIm = self.totalMaskedIm.copy()

        #If the Draw Contours option is selected, will draw contours on the displayed image
        if self.drawConts.get():
            self.drawContours()
            if len(self.numberTextArgs)>0:
                for i in range(len(self.numberTextArgs)):
                    cv2.putText(self.dispImDraw, *self.numberTextArgs[i][:-1], self.numberTextArgs[i][-1] if not self.showRefinedZones.get() else (255,255,255), thickness=10)
            self.base.displayCVImage(self.dispImDraw)
        else:
            self.base.displayCVImage(self.dispIm)


    #Draws contours onto the displayed image
    def drawContours(self):
        #Making a copy of the image so we don't lose the analysis steps when we draw
        self.dispImDraw = self.dispIm.copy()

        #If the user hasn't refined the zones, draw the contours
        if len(self.refinedMasks)==0:
            cv2.drawContours(self.dispImDraw, self.contours, -1, (255, 0, 255),thickness=3)
            self.inContour = -1

            #If the mouse is in the frame, check if it's in a contour
            if self.inFrame.get():
                for i in range(len(self.contours)):
                    contour = self.contours[i]
                    #The contours will be in order of size ascending, so the first contour the mouse is in
                    #   will be the smallest
                    if cv2.pointPolygonTest(contour, (self.mousex,self.mousey),False)>=0:
                        self.inContour = i
                        cv2.drawContours(self.dispImDraw, [self.contours[i]], -1, (0, 255,0), thickness=4)
                        break

            #If the user has clicked in a contour to select it, draw it in a different color
            if self.selectedCont!=-1:
                cv2.drawContours(self.dispImDraw, [self.contours[self.selectedCont]],
                                 -1, (0, 255, 255), thickness=5)

                #If we have found similar contours, draw them in a different color too
                if len(self.closeInds)!=0:
                    cv2.drawContours(self.dispImDraw, self.contours[self.closeInds],
                                     -1, (255, 255, 0), thickness=4)
                    #If the user has added contours, draw them in a slightly different color
                    if len(self.addConts)!=0:
                        cv2.drawContours(self.dispImDraw, self.contours[self.addConts],
                                         -1, (255, 128, 0), thickness=4)
                    if len(self.removeConts)!=0:
                        cv2.drawContours(self.dispImDraw, self.contours[self.removeConts],
                                         -1, (0, 0, 255), thickness=4)
        #If the user has refined the zones, draw the refined zones
        else:
            for i in range(len(self.refinedMasks)):
                center = self.refinedCenters[i].astype(int)
                drawShape(self.dispImDraw, self.zoneShape, center, self.refiner_data, (0,0,255), thickness=4)
                

            


    #Applies a thresholding mask to the image based on the present state of the sliders
    def cvMask(self, val=None):

        #Making sure the thresholds are integers between 0 and 255
        self.V_maskThresh1.set(np.clip(int(self.maskThresh1Slider.get()),0,255))
        self.V_maskThresh2.set(np.clip(int(self.maskThresh2Slider.get()),0,255))
        
        
        smin = int(self.V_maskThresh2.get())
        vmin = int(self.V_maskThresh1.get())

        #Using the HSV colorspace to mask for saturation and value
        hsvMin_s = np.array([0,smin,0])
        hsvMin_v = np.array([0,0,vmin])
        hsvMax = np.array([255,255,255])

        #The default mode is to AND the masks
        if self.V_maskMode.get()==0:
            self.analyzed = cv2.inRange(self.imHSV, hsvMin_s+hsvMin_v, hsvMax)
        
        #There is an option to OR them instead (slightly slower)
        elif self.V_maskMode.get()==1:
            self.mask_s = cv2.inRange(self.imHSV, hsvMin_s, hsvMax)
            self.mask_v = cv2.inRange(self.imHSV, hsvMin_v, hsvMax)
            self.analyzed = np.array(np.logical_or(self.mask_s, self.mask_v)*255, dtype=np.uint8)



    #Applies a series of dilations and erosions to the mask depending on the code
    #   Argument: code is a string of ['e','d'] of arbitrary length to indicate the
    #   order in which the user has pressed the Dilate and Erode buttons
    def cvDilateErode(self, code=None):
        self.dilateCounter.set(0)
        self.erodeCounter.set(0)
        dilerocode_text = self.V_dilerocode.get()
        for i in range(len(dilerocode_text)):
            code = dilerocode_text[i]
            if code=='d':
                self.analyzed = cv2.dilate(self.analyzed, (5,5))
                self.dilateCounter.set(self.dilateCounter.get()+1)
            if code=='e':
                self.analyzed = cv2.erode(self.analyzed, (5,5))
                self.erodeCounter.set(self.erodeCounter.get()+1)


    #Applies a blurring filter to the mask based on the present state of the slider
    def cvBlur(self, val=None):
        blur = np.clip(int(self.blurSlider.get()),0,10)
        self.V_blurAmount.set(blur)
        self.analyzed = cv2.blur(self.analyzed, (blur, blur))
        


    #Detects contours in the mask
    def cvContour(self):

        self.refinedMasks = []

        #If the image is in grayscale (only two coordinates, or third dimension is 1), find contours
        if len(self.analyzed.shape)==2 or self.analyzed.shape[-1]==1:

            #Finds contours in the image
            res = cv2.findContours(self.analyzed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            #Convert list of contours to an array
            #The if statement is necessary because of stupid compatibility reasons
            #(The return values of findContours changed between versions 3 and 4
            if cv2.__version__.startswith('3'):
                self.contours = np.array(res[1])
            else:
                self.contours = np.array(res[0])

                
            #Initializing the index of the user-selected contour
            self.selectedCont = -1

            #Initializing an array of sizes
            self.sizes = np.zeros(len(self.contours))

            #Initializing a list of text parameters for later printing
            self.numberTextArgs = []

            #Finding and storing the size of all the contours
            for i in range(len(self.contours)):
                c = self.contours[i]
                self.sizes[i] = cv2.contourArea(c)
                
            #Sorting the contours by size ascending
            self.bysize = np.argsort(self.sizes)
            #We only want to consider the reasonably sized contours, so I arbitrarily picked the ones
            #   with an area above 5 pixels --- this may cause problems.
            self.bysize = self.bysize[self.sizes[self.bysize]>5] 
            self.contours = self.contours[self.bysize]
            self.sizes = self.sizes[self.bysize]




            #Disable the Find Similar Contours button
            self.similarContsButton.state(["disabled"])


            #Disable the size tolerance slider
            self.sizeTolLabel.state(["disabled"])
            self.sizeTolSlider.state(["disabled"])
            self.sizeTolIndicator.state(["disabled"])

            #Disable the shape tolerance slider
            self.shapeTolLabel.state(["disabled"])
            self.shapeTolSlider.state(["disabled"])
            self.shapeTolIndicator.state(["disabled"])

            #Disable the Refine Zones and Analyze buttons
            self.refineButton.state(["disabled"])
            self.analyzeButton.state(["disabled"])

            #Disable the output options
            self.outputLabel.state(["disabled"])
            self.RGBCheck.state(["disabled"])
            self.HSVCheck.state(["disabled"])
            self.LABCheck.state(["disabled"])
            self.histCheck.state(["disabled"])




            #Enable the checkbutton option to display the contours
            self.showContCheck.state(['!disabled'])
            self.drawConts.set(True)

            #Start looking at the mouse to let the user pick a contour
            self.base.display.bind("<Motion>", self.trackMouse)
            #Left-click selects a contour
            self.base.display.bind("<Button-1>", self.selectContour)
            

            #Displays the number of contours found
            self.contourCount.config(text = "Found "+str(len(self.contours))+" contour"+("s" if len(self.contours)!=1 else ""))

            self.updateImage()

        #If the image is not grayscale the user needs to mask it
        else:
            print("Please mask your image first!")


    #Watches for the mouse and updates the position and image accordingly
    def trackMouse(self, event):
        newx, newy = self.convertCoords(event.x, event.y)
        self.mousex, self.mousey = newx, newy
        self.updateImage()

    #Waits for a mouse click and sets the index of the selected contour
    def selectContour(self, event):
        
        newx, newy = self.convertCoords(event.x, event.y)

        if self.inContour!=-1:
            #self.inContour is defined in updateImage when drawing the contours
            self.selectedCont = self.inContour

            #Selecting a new reference contour resets the arrays of similar contours
            self.closeInds = []
            self.addConts = []
            self.removeConts = []

            self.numberTextArgs = []


            #Enables the Find Similar Contours button
            self.similarContsButton.state(["!disabled"])


            #Disable the size tolerance slider
            self.sizeTolLabel.state(["disabled"])
            self.sizeTolSlider.state(["disabled"])
            self.sizeTolIndicator.state(["disabled"])

            #Disable the shape tolerance slider
            self.shapeTolLabel.state(["disabled"])
            self.shapeTolSlider.state(["disabled"])
            self.shapeTolIndicator.state(["disabled"])

            #Disable the Refine Zones and Analyze buttons
            self.refineButton.state(["disabled"])
            self.analyzeButton.state(["disabled"])

            #Disable the output options
            self.outputLabel.state(["disabled"])
            self.RGBCheck.state(["disabled"])
            self.HSVCheck.state(["disabled"])
            self.LABCheck.state(["disabled"])
            self.histCheck.state(["disabled"])



            self.selectedContLabel.config(text=(f"Selected contour {self.selectedCont}") if self.selectedCont>0 else "")
            self.contourCount.config(text=self.contourCount.cget("text").split(" | ")[0])
            self.updateImage()



    #Waits for a shift-click and adds the contour that the mouse is in to the list of added contours
    def appendContour(self, event):
        newx, newy = self.convertCoords(event.x, event.y)
        #If the user shift-clicks before similar contours are found it redirects to normal click
        if len(self.closeInds)==0:
            self.selectContour(event)
        #If there is a list of similar contours this will add to them
        else:
            #If the contour hasn't been added yet add it
            if self.inContour not in self.closeInds and self.inContour not in self.addConts:
                self.addConts.append(self.inContour)
            #If the contour has already been added remove it
            elif self.inContour not in self.closeInds and self.inContour in self.addConts:
                self.addConts.remove(self.inContour)
            #If the contour is already detected add it to the list to remove
            elif self.inContour in self.closeInds and self.inContour not in self.removeConts:
                self.removeConts.append(self.inContour)
            #If the contour is in the list to remove, remove it from that list
            elif self.inContour in self.closeInds and self.inContour in self.removeConts:
                self.removeConts.remove(self.inContour)

            #Getting the array of contours after manual adjustment
            self.closeIndsPlus = np.union1d(self.closeInds, self.addConts).astype(int)
            self.closeIndsPlus = np.setdiff1d(self.closeIndsPlus, self.removeConts)
            
            #Updating the contour count accordingly
            self.contourCount.config(text=self.contourCount.cget("text").split(" | ")[0]+f" | {len(self.closeInds)} similar"+\
                                     (f" + {len(self.addConts)}" if (len(self.addConts)>0) else "")+\
                                     (f" - {len(self.removeConts)}" if (len(self.removeConts)>0) else ""))

            #Finding the centers of the new contours
            self.findCenters()

            #Updates the image accordingly
            self.updateImage()

        
    #Takes mouse coordinates in the frame and converts them to pixel coordinates in the image
    def convertCoords(self, x, y):
        frameWidth, frameHeight = self.base.display.winfo_width(), self.base.display.winfo_height()
        imWidth, imHeight = self.im.shape[1], self.im.shape[0]

        Wratio = (imWidth-1)/(frameWidth-1)
        Hratio = (imHeight-1)/(frameHeight-1)

        return int(x*Wratio), int(y*Hratio)


    #Finds the contours that have a size and shape within a certain tolerance of the selected contour
    def getSimilarContours(self, val=None):

        #Enable the size tolerance slider
        self.sizeTolLabel.state(["!disabled"])
        self.sizeTolSlider.state(["!disabled"])
        self.sizeTolIndicator.state(["!disabled"])

        #Enable the shape tolerance slider
        self.shapeTolLabel.state(["!disabled"])
        self.shapeTolSlider.state(["!disabled"])
        self.shapeTolIndicator.state(["!disabled"])

        #Enable the Refine Zones and Analyze buttons
        self.refineButton.state(["!disabled"])
        self.analyzeButton.state(["!disabled"])

        #Enable the output options
        self.outputLabel.state(["!disabled"])
        self.RGBCheck.state(["!disabled"])
        self.HSVCheck.state(["!disabled"])
        self.LABCheck.state(["!disabled"])
        self.histCheck.state(["!disabled"])


        #Rounds the size tolerance (preset var) and makes sure it's between 0 and 100
        self.V_sizeTol.set(np.round(np.clip(self.V_sizeTol.get(),0,100),1))

        #Rounds the shape tolerance (preset var) and makes sure it's between 0 and 2
        self.V_shapeTol.set(np.round(np.clip(self.V_shapeTol.get(),0,2),3))


        #First eliminates contours by size
        self.closeInds = np.where(np.isclose(self.sizes, self.sizes[self.selectedCont], rtol=self.V_sizeTol.get()/100))[0]


        #Finds the shape-match score for each contour compared to the reference
        #   See https://docs.opencv.org/3.1.0/d3/dc0/group__imgproc__shape.html#gaadc90cb16e2362c9bd6e7363e6e4c317
        #   for more information
        shapeMatches = [cv2.matchShapes(self.contours[ind], self.contours[self.selectedCont], CV_CONTOURS_MATCH_I3, 0) for ind in self.closeInds]
        shapeMatches = np.array(shapeMatches)
        doesMatch = shapeMatches<self.V_shapeTol.get() #boolean array of where the shapes do match within tolerance
        self.closeInds = self.closeInds[doesMatch]

        #Resetting the removed contours because changing the thresholds could result in removing contours that aren't there
        self.removeConts = []

        self.closeIndsPlus = np.union1d(self.closeInds, self.addConts).astype(int)


        #Shift-left-click allows user to add a contour to the list
        self.base.display.bind("<Shift-Button-1>", self.appendContour)


        self.contourCount.config(text=self.contourCount.cget("text").split(" | ")[0]+" | "+\
                                 str(len(self.closeInds))+" similar"+(f" + {len(self.addConts)}" if len(self.addConts)>0 else ""))
        self.findCenters()
        self.updateImage()


    #Finds the center of each contour
    def findCenters(self):

        #Initializing lists of center coordinates and sizes
        self.centers = np.zeros((len(self.closeIndsPlus),2))
        self.closeSizes = np.zeros((len(self.closeIndsPlus)))

        #A dictionary that will allow us to go from indices in the contour array to
        #   indices in the centers array
        self.indDict = {}

        #Calculates center and size for the contours in the closeIndsPlus array
        for i in range(len(self.closeIndsPlus)):
            ind = int(self.closeIndsPlus[i])
            cont = self.contours[ind]

            self.indDict[ind] = i

            #Computing moments of the contour
            M = cv2.moments(cont)

            #Central moments are given by M_10/M_00 and M_01/M_00,
            #   where M_ij = sum(x^i y^j I(x,y)) over the whole image,
            #   where I is either 1 or 0
            #   Finds the centroid of the image
            self.centers[i] = np.array([int(M['m10']/M['m00']),int(M['m01']/M['m00'])]) #x,y

            #For reference: M_00 is sum(I(x,y)) 
            self.closeSizes[i] = M['m00'] #0,0th moment is the sum of the pixels in the image



    #Returns the average color and standard deviation of an image,
    #   if a mask is given returns the average color of the masked area
    def getAvColor(self, im, mask=None):
        if mask is not None:
            mask3D = np.concatenate(([mask],[mask],[mask])).transpose((1,2,0))/255 #triplicating the mask values for RGB etc.
            immasked = np.ma.MaskedArray(im, mask=1-mask3D)
            avcolor = np.ma.mean(immasked, axis=(0,1))
            std = np.ma.std(immasked, axis=(0,1)) #standard deviation

        else:
            avcolor = np.average(np.average(im, axis=0), axis=0)
            std = np.std(im, axis=(0,1))
            
        return avcolor, std


    #Creates a zone refinement dialog and sets refined zones based on user input
    #Triggered by Refine Zones button
    def refineZones(self):
        #Finding the boundaries of the selected contour (the one to be used as a reference for zone refinement
        x, y, w, h = cv2.boundingRect(self.contours[self.selectedCont])

        #Cropping the image to this region, with some padding
        zoneCrop = self.im[y-self.saveBorder:y+h+self.saveBorder,\
                                  x-self.saveBorder:x+w+self.saveBorder]

        #Shift center coordinates from image coordinates to crop coordinates
        center = self.centers.astype(int)[self.indDict[self.selectedCont]] - (np.array([x,y]) - self.saveBorder)


        
        #Setting up the tk window for the refinement interface
        self.refinerWindow = tk.Toplevel(master=self.window)

        #Creating and running a ZoneRefiner object (see class definition)
        self.refiner = ZoneRefiner(self.refinerWindow, zoneCrop, center)

        #The program will wait until the user closes the refiner window
        self.window.wait_window(self.refinerWindow)


        #Extracting the information from the refiner object
        self.zoneShape, self.displace_x, self.displace_y, self.refiner_data = self.refiner.getParams()


        #Initializing variables to store masks and new zone centers
        self.refinedMasks = np.zeros((len(self.centers),*self.im.shape[:2]), dtype=np.uint8)
        self.refinedCenters = np.zeros_like(self.centers)
        mask = np.zeros(self.im.shape[:2], dtype=np.uint8)

        #Applying the displacement to contour centers and making a set of masks for each refined zone
        for i in range(len(self.centers)):
            mask_i = np.zeros(self.im.shape[:2], dtype=np.uint8)
            self.refinedCenters[i] = self.centers[i]+np.array([self.displace_x, -self.displace_y])

            #Making a total mask for display purposes
            drawShape(mask, self.zoneShape, self.refinedCenters[i].astype(int), self.refiner_data, color=255, thickness=-1)

            #Making individual masks for each contour for analysis
            drawShape(mask_i, self.zoneShape, self.refinedCenters[i].astype(int), self.refiner_data, color=255, thickness=-1)
            self.refinedMasks[i] = mask_i[:,:]

        #Enabling the Show Masks check button
        self.refineMaskShowCheck.state(['!disabled'])

        self.totalMaskedIm = cv2.bitwise_and(self.im, self.im, mask=mask)
        self.totalMask = mask

        #Will now draw the refined zones on the image
        self.updateImage()

        

    #Final contour analysis
    #Triggered by Analysis button
    def analyzeContours(self):
        print("ANALYZING")

        #Initializing arrays to store the zone color information
        self.avcolorsRGB = np.zeros((len(self.closeIndsPlus), 3))
        self.avcolorsHSV = np.zeros((len(self.closeIndsPlus), 3))
        self.avcolorsLAB = np.zeros((len(self.closeIndsPlus), 3))
        self.stdsRGB = np.zeros((len(self.closeIndsPlus), 3))
        self.stdsHSV = np.zeros((len(self.closeIndsPlus), 3))
        self.stdsLAB = np.zeros((len(self.closeIndsPlus), 3))
        self.maskAreas = np.zeros(len(self.closeIndsPlus))

        #Array to store position, size, etc. for text to place on image
        self.numberTextArgs = np.zeros(len(self.closeIndsPlus), dtype=object)


        #Making a unique folder name for this analysis output
        analysisPath = os.path.splitext(self.base.filePath)[0]+'_analysis'
        foldernum = 0
        self.analysisPathNum = analysisPath
        while os.path.exists(self.analysisPathNum):
            foldernum+=1
            self.analysisPathNum = analysisPath+'_'+str(foldernum)
        os.makedirs(self.analysisPathNum)
        
        
        #Various aesthetic properties will be decided by the size of the largest contour
        #   Note: this will behave poorly when the analyzed contours are of very different sizes
        largestContInd = self.closeIndsPlus[np.argmax(self.closeSizes)]
        largest_x, largest_y, largest_w, largest_h = cv2.boundingRect(self.contours[largestContInd])


        #Sorting indices first by row then by column
        sort_inds = np.lexsort((self.centers[:,0],self.centers[:,1]))

        #Sorting the arrays
        self.closeIndsPlus = self.closeIndsPlus[sort_inds]
        self.centers = self.centers[sort_inds]

        #If the user has refined the zones, sorting those arrays too
        if len(self.refinedMasks)!=0:
            self.refinedMasks = self.refinedMasks[sort_inds]
            self.refinedCenters = self.refinedCenters[sort_inds]


        #Looping through all the close contours to analyze them
        for i in range(len(self.closeIndsPlus)):
            
            ind = self.closeIndsPlus[i]
            cont = self.contours[ind]

            cont_x, cont_y, cont_w, cont_h = cv2.boundingRect(cont)


            #If there are no refined masks (the user has not refined zones)
            #   then use contours
            if len(self.refinedMasks)==0:
                x, y, w, h = cont_x, cont_y, cont_w, cont_h
                contMask = np.zeros((self.im.shape[0],self.im.shape[1]), dtype=np.uint8)
                cv2.drawContours(contMask, [cont], -1, 255, thickness=-1)
                center = self.centers[i]

                self.totalMask = np.bitwise_or(contMask,self.totalMask)

            #If the user has refined zones, use the refined zones
            else:
                contMask = self.refinedMasks[i]
                center = self.refinedCenters[i]
                x = int(center[0]-self.refiner_data[-1][0])
                y = int(center[1]-self.refiner_data[-1][-1])
                w = self.refiner_data[-1][0]*2+1
                h = self.refiner_data[-1][-1]*2+1
                


            #Getting average color of the zone and standard deviation
            avcolorRGB, stdRGB = self.getAvColor(self.im[y:y+h,x:x+w], contMask[y:y+h,x:x+w])
            avcolorRGB = avcolorRGB[::-1]#reversing because opencv uses BGR
            stdRGB = stdRGB[::-1]#ditto

            #Doing the same in the other color spaces
            avcolorHSV, stdHSV = self.getAvColor(self.imHSV[y:y+h,x:x+w], contMask[y:y+h,x:x+w])
            avcolorLAB, stdLAB = self.getAvColor(self.imLAB[y:y+h,x:x+w], contMask[y:y+h,x:x+w])

            #Sticking them in the list
            self.avcolorsRGB[i, :] = avcolorRGB
            self.avcolorsHSV[i, :] = avcolorHSV
            self.avcolorsLAB[i, :] = avcolorLAB
            self.stdsRGB[i, :] = stdRGB
            self.stdsHSV[i, :] = stdHSV
            self.stdsLAB[i, :] = stdLAB

            #Getting the area of the masked region
            self.maskAreas[i] = np.sum(contMask[y:y+h,x:x+w])/255

            #Finding a referencee text size for scaling
            t_size, baseline = cv2.getTextSize(str(i+1), cv2.FONT_HERSHEY_SIMPLEX, 1, 10)

            #Scaling by the size of the largest contour
            #TODO: Find a better way to scale the text!!
            fontsize = min(int(np.round(largest_w/t_size[0])),int(np.round(largest_h/t_size[1])))
            
            #Setting the position of the number to the top right of the contour
            textcent = (int(center[0]+w//2), int(center[1]-h//2))
            color = [0,255,0]
                
            self.numberTextArgs[i]=(str(i+1), textcent, cv2.FONT_HERSHEY_SIMPLEX, fontsize, color)
            


            print(f'Spot {i+1} analyzed')


            #If the user has elected to save the histograms (slow!)
            if self.V_saveHistograms.get():
                histspath = self.analysisPathNum+'/histograms'
                if not os.path.exists(histspath):
                    os.makedirs(histspath)
                self.saveHistogram(self.im[y:y+h,x:x+w], contMask[y:y+h,x:x+w], path=histspath+'/'+self.base.filename+'_histogram_'+str(i+1))

            #Saves an image cropped to the current zone
            if self.saveCrops:
                cropspath = self.analysisPathNum+'/crops'
                if not os.path.exists(cropspath):
                    os.makedirs(cropspath)
                    os.makedirs(cropspath+'/drawn')

                #Takes a slice of the image from the top-left corner of the zone
                #   with the dimensions of the largest of the close contours (for consistent crop sizes)    
                crop_im = self.im[cont_y-self.saveBorder:cont_y+largest_h+self.saveBorder,\
                                  cont_x-self.saveBorder:cont_x+largest_w+self.saveBorder]

                #A copy of the original image to draw just the contour on
                crop_im_draw = self.im.copy()

                #Drawing either the contour or the refined zone shape
                if len(self.refinedMasks)==0:
                    cv2.drawContours(crop_im_draw, [cont], -1, (255,255,0), 1)
                else:
                    drawShape(crop_im_draw, self.zoneShape, self.refinedCenters[i], self.refiner_data, (0,0,255),1)

                #Cropping that result down (we do it in this order so we don't have to transform any coordinates)
                crop_im_draw = crop_im_draw[cont_y-self.saveBorder:cont_y+largest_h+self.saveBorder,\
                              cont_x-self.saveBorder:cont_x+largest_w+self.saveBorder]

                #Saving the resulting images               
                cv2.imwrite(cropspath+'/'+self.base.filename+'_crop_'+str(i+1)+'.jpg', crop_im)
                cv2.imwrite(cropspath+'/drawn/'+self.base.filename+'_crop_draw_'+str(i+1)+'.jpg', crop_im_draw)

        #Draws the numbers on the screen
        self.updateImage()

        #Saves the average colors to a csv file
        self.saveColors()

        #Saves the numbered image and the mask for reference
        self.saveIm()

            
    #Saving the average colors to a csv file
    def saveColors(self):

        #Making a column of numbers in ascending order to match numbers drawn on image
        labelcol = np.arange(len(self.closeIndsPlus))+1

        #A blank column which will be used to space colorspaces
        spacer = np.array([['']*len(labelcol)]).T

        #Initializing the array to be filled with the colors
        #   (We will append columns each time)
        full = np.array([labelcol]).T

        #Can't be capitalized -- Excel interprets that weirdly
        header = 'id'

        #If the user has elected to save RGB colors 
        if self.V_saveRGB.get():
            rgb = self.avcolorsRGB
            std_rgb = self.stdsRGB
            header+=',R,G,B,std R,std G,std B,'
            rgb_s = rgb.astype(str)
            std_rgb_s = std_rgb.astype(str)
            full = np.concatenate((full, rgb_s, std_rgb_s, spacer), axis=1)

        #The program will always output at least the grayscale
        #Grayscale values calculated as a weighted average of RGB
        grayscale = np.dot(rgb, RGB2grayscale_weights)
        std_grayscale = np.sqrt(np.dot(std_rgb**2, RGB2grayscale_weights**2)) #sqrt of sum of squares for proper propagation of error
        grayscale_s = np.array([grayscale.astype(str)]).T
        std_grayscale_s = np.array([std_grayscale.astype(str)]).T
        header+=',Gray,std Gray,'
        full = np.concatenate((full, grayscale_s, std_grayscale_s, spacer), axis=1)


        #If the user has elected to save HSV colors
        if self.V_saveHSV.get():
            hsv = self.avcolorsHSV
            hsv[:,0] = hsv[:,0]/180*360
            hsv[:,1:] = np.round(hsv[:,1:]/255, 8)
            std_hsv = self.stdsHSV
            std_hsv[:,0] = std_hsv[:,0]/180*360
            std_hsv[:,1:] = np.round(std_hsv[:,1:]/255, 8)
            header+=',H,S,V,std H,std S,std V,'
            hsv_s = hsv.astype(str)
            std_hsv_s = std_hsv.astype(str)
            full = np.concatenate((full, hsv_s, std_hsv_s, spacer), axis=1)

        #If the user has elected to save Lab colors
        if self.V_saveLAB.get():
            lab = self.avcolorsLAB
            lab[:,0] = np.round(lab[:,0]/255*100, 8)
            lab[:,1:] = lab[:,1:]-128
            std_lab = self.stdsLAB
            std_lab[:,0] = np.round(std_lab[:,0]/255*100, 8)
            header+=',L,a,b,std L,std a,std b,'
            lab_s = lab.astype(str)
            std_lab_s = std_lab.astype(str)
            full = np.concatenate((full, lab_s, std_lab_s, spacer), axis=1)


        #The program will always output the contour areas (or refined zone areas)
        area_s = np.array([self.maskAreas.astype(str)]).T
        full = np.concatenate((full, area_s), axis=1)
        header+=',Area [pixels]'

        #Saving the full output in a csv format
        np.savetxt(self.analysisPathNum+'/'+self.base.filename+"_colors.csv", full, delimiter=',', header = header, fmt='%s', comments='')


    #Saving the image with numbers drawn on
    def saveIm(self):
        #copying the image so we don't edit the original
        imcopy = self.im.copy()

        #Drawing the numbers on
        for i in range(len(self.numberTextArgs)):
            cv2.putText(imcopy, *self.numberTextArgs[i], thickness=10)

        #If the user has refined the zones, draw the zones
        if len(self.refinedMasks)!=0:
            for i in range(len(self.refinedMasks)):
                center = self.refinedCenters[i].astype(int)
                drawShape(imcopy, self.zoneShape, center, self.refiner_data, color=(0,0,255), thickness=4)
        #Otherwise, draw the contours
        else:
            cv2.drawContours(imcopy, self.contours[self.closeIndsPlus], -1, (255,255,0), 4)

        #Save the labeled image and the mask
        cv2.imwrite(self.analysisPathNum+'/'+self.base.filename+"_labeled"+self.base.ext, imcopy)
        cv2.imwrite(self.analysisPathNum+'/'+self.base.filename+"_mask"+self.base.ext, self.totalMask)
                

    #Saving a histogram for an image
    #Triggered by the Analyze button if the user has elected to save the histograms
    def saveHistogram(self, im, mask, path):

        #Computing the histograms for each channel
        Bheights, edges = imageChannelHistogram(im[:,:,0], mask)
        Gheights, edges = imageChannelHistogram(im[:,:,1], mask)
        Rheights, edges = imageChannelHistogram(im[:,:,2], mask)

        #Converting the result to strings for saving as a csv
        edges_str = edges.astype(int).astype(str)[:-1]
        Bheights_str = Bheights.astype(int).astype(str)
        Gheights_str = Gheights.astype(int).astype(str)
        Rheights_str = Rheights.astype(int).astype(str)

        #Sticking them together
        combo = np.concatenate(([edges_str], [Rheights_str], [Gheights_str], [Bheights_str]), axis=0).T

        #Saving the csv file
        np.savetxt(path+'.csv', combo, fmt='%s', header='bin, Red Channel, Green Channel, Blue Channel', delimiter=',', comments='')

        ###Also saving plots of the histograms for immediate inspection###

        #Average of successive edges will be the centers of the bins
        centers = (edges[:-1] + edges[1:]) / 2


        #Setup for histogram plotting
        histoFig = plt.figure()
        histoAxis = histoFig.add_subplot(111)

        histoAxis.set_facecolor('xkcd:grey')
        histoAxis.set_xlim([0,256])
        histoAxis.set_xticks(np.linspace(0,256,9))
        histoAxis.set_xlabel("Intensity")
        histoAxis.set_ylabel("Counts")
        histoAxis.set_title(f"Zone {path.split('_')[-1]} Histogram")


        #Plots the histograms
        histoPlotBlue = histoAxis.bar(centers, Bheights, align='center', color='blue', width=edges[1] - edges[0], alpha=0.6)
        histoPlotGreen = histoAxis.bar(centers, Gheights, align='center', color='green', width=edges[1] - edges[0], alpha=0.6)
        histoPlotRed = histoAxis.bar(centers, Rheights, align='center', color='red', width=edges[1] - edges[0], alpha=0.6)

        #Uncomment for Photoshop-style color-mixing intersections in the saved histograms
##        histoPlotCyan = histoAxis.bar(centers, np.min([Bheights, Gheights],axis=0), align='center', color='cyan', width=edges[1] - edges[0], alpha=1)
##        histoPlotMagenta = histoAxis.bar(centers, np.min([Bheights, Rheights],axis=0), align='center', color='magenta', width=edges[1] - edges[0], alpha=1)
##        histoPlotYellow = histoAxis.bar(centers, np.min([Rheights, Gheights],axis=0), align='center', color='yellow', width=edges[1] - edges[0], alpha=1)
##
##        histoPlotWhite = histoAxis.bar(centers, np.min([Bheights, Gheights, Rheights],axis=0), align='center', color='white', width=edges[1] - edges[0], alpha=1)

        #Saving the figures
        histoFig.savefig(path+'.png')
##        plt.close()



#Object: ZoneRefiner
#Purpose: Interface for refining the zone shape, size, and position
class ZoneRefiner:

    def __init__(self, window, im, center):

        self.window = window
        window.title("Refine Zone")


        self.im = im

        self.h, self.w = self.im.shape[:2]

        #Calculating the proper length to make the entry boxes
        #   The longest string entered will probably be the same number of digits as the size
        #   (With a minus sign for one more)
        entryLength = max(len(str(-self.w//2)),len(str(-self.h//2)))+1
        
        #Making sure the center is an integer value for drawing
        self.center = center.astype(int)



        ####Histogram####

        #Setting up histogram plotting
        self.histoFig = Figure()
        self.histoAxis = self.histoFig.add_subplot(111)
        self.histoAxis.set_facecolor('xkcd:grey')
        self.histoAxis.set_xlim([0,256])
        self.histoAxis.set_xticks(np.linspace(0,256,9))
        self.histoAxis.set_xlabel("Intensity")
        self.histoAxis.set_ylabel("Counts")
        self.histoAxis.set_title("Zone Histogram")

        #Making a new window to put the histogram in
        self.histoWindow = tk.Toplevel(self.window)
        self.histoWindow.title("Zone Histogram")

        self.histoWindow.grid_columnconfigure(0, weight=1)
        self.histoWindow.grid_rowconfigure(0, weight=1)

        #Hiding the histogram window until it is requested
        self.histoWindow.withdraw()

        #To stick the histogram plot into a tk window we need to set up the canvas as a tk object
        self.histoCanvas = FigureCanvasTkAgg(self.histoFig, master=self.histoWindow)
        self.histoCanvas.draw()
        self.histoCanvas.get_tk_widget().grid(column=0,row=0, sticky='nesw')


        #Getting the heights and edges of the histogram
        #   (Filled with blank data initially)
        self.blankHeights, edges = imageChannelHistogram(-1*np.ones((1,1,1)))
        self.centers = (edges[:-1] + edges[1:]) / 2

        #Initializing plots with blank histograms
        self.histoPlotBlue = self.histoAxis.bar(self.centers, self.blankHeights, align='center', color='blue', width=edges[1] - edges[0], alpha=0.6)
        self.histoPlotGreen = self.histoAxis.bar(self.centers, self.blankHeights, align='center', color='green', width=edges[1] - edges[0], alpha=0.6)
        self.histoPlotRed = self.histoAxis.bar(self.centers, self.blankHeights, align='center', color='red', width=edges[1] - edges[0], alpha=0.6)

        self.histoPlotCyan = self.histoAxis.bar(self.centers, self.blankHeights, align='center', color='cyan', width=edges[1] - edges[0], alpha=1)
        self.histoPlotMagenta = self.histoAxis.bar(self.centers, self.blankHeights, align='center', color='magenta', width=edges[1] - edges[0], alpha=1)
        self.histoPlotYellow = self.histoAxis.bar(self.centers, self.blankHeights, align='center', color='yellow', width=edges[1] - edges[0], alpha=1)

        self.histoPlotWhite = self.histoAxis.bar(self.centers, self.blankHeights, align='center', color='white', width=edges[1] - edges[0], alpha=1)



        #A frame in the histogram window for channel display checkbuttons
        self.histoFrame = ttk.Frame(master=self.histoWindow)
        self.histoFrame.grid(column=1,row=0)

        #Checkbuttons for each histogram channel
        self.plotHistoRed = tk.BooleanVar(value=True)
        self.plotHistoRedCheck = ttk.Checkbutton(master=self.histoFrame, text="Red Channel",
                                        variable=self.plotHistoRed, onvalue=True, command=self.plotHistogram)
        self.plotHistoRedCheck.grid(sticky='w')

        self.plotHistoGreen = tk.BooleanVar(value=False)
        self.plotHistoGreenCheck = ttk.Checkbutton(master=self.histoFrame, text="Green Channel",
                                        variable=self.plotHistoGreen, onvalue=True, command=self.plotHistogram)
        self.plotHistoGreenCheck.grid(sticky='w')

        self.plotHistoBlue = tk.BooleanVar(value=False)
        self.plotHistoBlueCheck = ttk.Checkbutton(master=self.histoFrame, text="Blue Channel",
                                        variable=self.plotHistoBlue, onvalue=True, command=self.plotHistogram)
        self.plotHistoBlueCheck.grid(sticky='w')


        
        spacer = ttk.Label(master=self.histoFrame, text="")
        spacer.grid()


        
        #A checkbutton to display the intersections between colors as color mixing
        #   (How it is done in Photoshop)
        self.plotHistoIntersection = tk.BooleanVar(value=False)
        self.plotHistoIntersectionCheck = ttk.Checkbutton(master=self.histoFrame, text="Intersections",
                                        variable=self.plotHistoIntersection, onvalue=True, command=self.plotHistogram)
        self.plotHistoIntersectionCheck.grid(sticky='w')



        ####Refiner####


        #Setting up image frame and menu frame        
        self.frame = ttk.Frame(self.window, borderwidth=5, relief="sunken")
        self.menu = ttk.Frame(self.window)
        self.frame.grid(column=0, row=0, sticky='nesw')
        self.menu.grid(column=1, row=0, sticky='nesw')

        #Making the firt row/column scalable scalable (image will scale)
        self.window.grid_rowconfigure(0, weight=1)
        self.window.grid_columnconfigure(0, weight=1)

        #Making the image frame scalable
        self.frame.grid_columnconfigure(0, weight=1)
        self.frame.grid_rowconfigure(0, weight=1)


        #Image display
        self.display = ttk.Label(self.frame, text="No Image Selected")
        self.display.grid(row=0, column=0)

        ####Setting up the menu####

        #Starting at row 0
        row = 0

        #Initial values for parameters
        #TODO: Make these preset variables
        radius_preset = self.w//4
        displace_x_preset = 0
        displace_y_preset = 0

        #Frame for the shape selection radiobuttons
        self.selectorFrame = ttk.Frame(self.menu)
        self.selectorFrame.grid(row=row, column=0, columnspan=4, sticky='we')
        
        self.selectorText = ttk.Label(self.selectorFrame, text='Select Zone Shape')
        self.selectorText.grid(columnspan=2, sticky='we')
        
        self.zoneShape = tk.StringVar(value='circle')

        #Radiobuttons for the currently supported shapes
        self.selectCircle = tk.Radiobutton(self.selectorFrame, text = "circle", value="circle",
                                                 variable=self.zoneShape, indicatoron=False, command=self.swap_menu)
        self.selectRectangle = tk.Radiobutton(self.selectorFrame, text = "rectangle", value="rectangle",
                                                 variable=self.zoneShape, indicatoron=False, command=self.swap_menu)
        self.selectPolygon = tk.Radiobutton(self.selectorFrame, text = "polygon", value="polygon",
                                                 variable=self.zoneShape, indicatoron=False, command=self.swap_menu)

        
        self.selectCircle.grid(row=1, column=0, sticky='we')
        self.selectRectangle.grid(row=1, column=1, sticky='we')
        self.selectPolygon.grid(row=1, column=2, sticky='we')


        row+=1
        ###Next Row###


        #Setting up slider for x displacement
        self.displace_x = tk.IntVar(self.window, value=displace_x_preset)

        self.x_label = ttk.Label(self.menu, text="X Displacement:")
        self.x_label.grid(row=row, column=0, sticky='e')
        self.x_slider = ttk.Scale(self.menu, from_=-self.w//2, to=self.w//2, orient=tk.HORIZONTAL, command=self.zoneDraw, variable=self.displace_x)
        self.x_slider.grid(row=row, column=1)
        self.x_indicator = ttk.Entry(self.menu, textvariable=self.displace_x, width=entryLength)
        self.x_indicator.grid(row=row, column=2)

        self.x_indicator.bind("<Key>", lambda e: self.zoneDraw() if e.keycode==TAB or e.keycode==ENTER else 0)


        row += 1
        ###Next Row###


        #Setting up slider for y displacement
        self.displace_y = tk.IntVar(self.window, value=displace_y_preset)

        self.y_label = ttk.Label(self.menu, text="Y Displacement:")
        self.y_label.grid(row=row, column=0, sticky='e')
        self.y_slider = ttk.Scale(self.menu, from_=-self.h//2, to=self.h//2, orient=tk.HORIZONTAL, command=self.zoneDraw, variable=self.displace_y)
        self.y_slider.grid(row=row, column=1)
        self.y_indicator = ttk.Entry(self.menu, textvariable=self.displace_y, width=entryLength)
        self.y_indicator.grid(row=row, column=2)

        self.y_indicator.bind("<Key>", lambda e: self.zoneDraw() if e.keycode==TAB or e.keycode==ENTER else 0)

    




        row += 1
        ###Next Row###


        #Setting up sliders for other shapes (some hidden initially)        

        #######For Circle##########

        #Setting up slider for circle/polygon radius
        
        self.radius = tk.IntVar(self.window, value=radius_preset)

        #To make the sliders hideable, we make their master the window, and grid them in the menu frame
        #   (with the in_ argument). To hide/show we use .lift() and .lower()
        self.r_label = ttk.Label(self.window, text="Radius:")
        self.r_label.grid(row=row, column=0, sticky='e', in_=self.menu)
        self.r_slider = ttk.Scale(self.window, from_=1, to=min(self.w//2,self.h//2), orient=tk.HORIZONTAL, command=self.zoneDraw, variable=self.radius)
        self.r_slider.grid(row=row, column=1, sticky='we', in_=self.menu)
        self.r_indicator = ttk.Entry(self.window, textvariable=self.radius, width=entryLength)
        self.r_indicator.grid(row=row, column=2, in_=self.menu)

        self.r_indicator.bind("<Key>", lambda e: self.zoneDraw() if e.keycode==TAB or e.keycode==ENTER else 0)


        #######For Rectangle##########

        #Setting up sliders for rectangle height and width
        self.rectangle_width = tk.IntVar(self.window, value=radius_preset)
        self.rectangle_height = tk.IntVar(self.window, value=radius_preset)
        
        self.width_label = ttk.Label(self.window, text="Width:")
        self.height_label = ttk.Label(self.window, text="Height:")
        self.width_label.grid(row=row, column=0, sticky='e', in_=self.menu)
        self.height_label.grid(row=row+1, column=0, sticky='e', in_=self.menu)
        self.width_slider = ttk.Scale(self.window, from_=1, to=self.w, orient=tk.HORIZONTAL, command=self.zoneDraw, variable=self.rectangle_width)
        self.height_slider = ttk.Scale(self.window, from_=1, to=self.h, orient=tk.HORIZONTAL, command=self.zoneDraw, variable=self.rectangle_height)
        self.width_slider.grid(row=row, column=1, sticky='we', in_=self.menu)
        self.height_slider.grid(row=row+1, column=1, sticky='we', in_=self.menu)
        self.width_indicator = ttk.Entry(self.window, textvariable=self.rectangle_width, width=entryLength)
        self.height_indicator = ttk.Entry(self.window, textvariable=self.rectangle_height, width=entryLength)
        self.width_indicator.grid(row=row, column=2, sticky='we', in_=self.menu)
        self.height_indicator.grid(row=row+1, column=2, sticky='we', in_=self.menu)

        self.width_indicator.bind("<Key>", lambda e: self.zoneDraw() if e.keycode==TAB or e.keycode==ENTER else 0)
        self.height_indicator.bind("<Key>", lambda e: self.zoneDraw() if e.keycode==TAB or e.keycode==ENTER else 0)


        self.width_label.lower(self.menu)
        self.height_label.lower(self.menu)
        self.width_slider.lower(self.menu)
        self.height_slider.lower(self.menu)
        self.width_indicator.lower(self.menu)
        self.height_indicator.lower(self.menu)
        


        #######For Polygons##########

        #Setting up slider for the number of sides of the polygon
        self.nsides = tk.IntVar(self.window, value=3)
        
        self.sides_label = ttk.Label(self.window, text="# Sides:")
        self.sides_label.grid(row=row+1, column=0, sticky='e', in_=self.menu)
        self.sides_slider = ttk.Scale(self.window, from_=3, to=6, orient=tk.HORIZONTAL, command=self.zoneDraw, variable=self.nsides)
        self.sides_slider.grid(row=row+1, column=1, sticky='we', in_=self.menu)
        self.sides_indicator = ttk.Entry(self.window, textvariable=self.nsides, width=entryLength)
        self.sides_indicator.grid(row=row+1, column=2, in_=self.menu)

        self.sides_indicator.bind("<Key>", lambda e: self.zoneDraw() if e.keycode==TAB or e.keycode==ENTER else 0)


        self.sides_label.lower(self.menu)
        self.sides_slider.lower(self.menu)
        self.sides_indicator.lower(self.menu)


        #Setting up slider for the angle at which the polygon is placed
        self.poly_angle = tk.DoubleVar(self.window, value=0)
        
        self.angle_label = ttk.Label(self.window, text="Angle:")
        self.angle_label.grid(row=row+2, column=0, sticky='e', in_=self.menu)
        #The maximum/minimum angle will be determined by the number of sides (regular polygons have n-rotational symmetry)
        self.angle_slider = ttk.Scale(self.window, from_=-180/self.nsides.get(), to=180/self.nsides.get(), orient=tk.HORIZONTAL, command=self.zoneDraw, variable=self.poly_angle)
        self.angle_slider.grid(row=row+2, column=1, sticky='we', in_=self.menu)
        self.angle_indicator = ttk.Entry(self.window, textvariable=self.poly_angle, width=entryLength)
        self.angle_indicator.grid(row=row+2, column=2, in_=self.menu)

        self.angle_indicator.bind("<Key>", lambda e: self.zoneDraw() if e.keycode==TAB or e.keycode==ENTER else 0)


        self.angle_label.lower(self.menu)
        self.angle_slider.lower(self.menu)
        self.angle_indicator.lower(self.menu)



        
        row += 2 #extra rows for extra hidden sliders
        row += 1
        ###Next Row###

        #Setting up a display for the shape area
        self.area = tk.IntVar(self.window, value=0)

        self.area_label = ttk.Label(self.menu, text="Area:")
        self.area_label.grid(row=row, column=1, sticky='e')
        self.area_indicator = ttk.Label(self.menu, textvariable=self.area)
        self.area_indicator.grid(row=row, column=2,sticky='we')

        row+=1
        ###Next Row###


        spacer = ttk.Label(master=self.menu, text="")
        spacer.grid(row=row)

        self.menu.grid_rowconfigure(row, weight=1)
        

        row += 1
        ###Next Row###


        #Checkbutton to mask the zone (display only the part on the image in the mask)
        self.maskZone = tk.BooleanVar(value=False)
        self.maskCheck = ttk.Checkbutton(self.menu, text="Mask Zone", variable=self.maskZone, \
                                         onvalue=True, command=self.zoneDraw)
        self.maskCheck.grid(row=row, column=0, sticky='e')

        #Checkbutton to display the histogram
        self.showHist = tk.BooleanVar(value=False)
        self.histCheck = ttk.Checkbutton(self.menu, text="Histogram", variable=self.showHist, \
                                         onvalue=True, command=self.showHideHistogram)
        self.histCheck.grid(row=row, column=1, sticky='e')


        #Makes it so the histogram can either be closed with checkbox or X button
        self.histoWindow.protocol("WM_DELETE_WINDOW", \
                                  lambda: [self.showHist.set(False), self.showHideHistogram()])



        row += 1
        ###Next Row###


        spacer = ttk.Label(master=self.menu, text="")
        spacer.grid(row=row)

        self.menu.grid_rowconfigure(row, weight=1)


        row += 1
        ###Next Row###


        #Closes window and allows AnalysisWindow to continue
        self.doneButton = ttk.Button(self.menu, text="Done Refining", command=self.window.destroy)
        self.doneButton.grid(row=row, column=0, columnspan=3, rowspan=2, sticky='nesw')



        self.menu.update()
        self.window.geometry(str(self.window.winfo_width())+'x'+str(self.window.winfo_height())+"+0+0")

        #Setting grid size propagation to false to prevent resizing feedback from image
        self.window.grid_propagate(False)

        self.displayCVImage(self.im, self.im.shape[:2])
        self.frame.bind("<Configure>", self.resize)


        self.zoneDraw()

    #Shows or hides the histogram when the checkbutton is pressed
    def showHideHistogram(self, event=None):
        if self.showHist.get():
            self.histoWindow.deiconify()
            self.plotHistogram()
        else:
            self.histoWindow.withdraw()


    #Swaps the menu between different zone shapes
    def swap_menu(self, event=None):

        ##Lowering all widgets##
        self.r_label.lower(self.menu)
        self.r_slider.lower(self.menu)
        self.r_indicator.lower(self.menu)

        self.width_label.lower(self.menu)
        self.height_label.lower(self.menu)
        self.width_slider.lower(self.menu)
        self.height_slider.lower(self.menu)
        self.width_indicator.lower(self.menu)
        self.height_indicator.lower(self.menu)

        self.sides_label.lower(self.menu)
        self.sides_slider.lower(self.menu)
        self.sides_indicator.lower(self.menu)
        self.angle_label.lower(self.menu)
        self.angle_slider.lower(self.menu)
        self.angle_indicator.lower(self.menu)
        

        ##Lifting the selected ones##
        if self.zoneShape.get()=='circle':
            self.r_label.lift(self.menu)
            self.r_slider.lift(self.menu)
            self.r_indicator.lift(self.menu)

        elif self.zoneShape.get()=='rectangle':
            self.width_label.lift(self.menu)
            self.height_label.lift(self.menu)
            self.width_slider.lift(self.menu)
            self.height_slider.lift(self.menu)
            self.width_indicator.lift(self.menu)
            self.height_indicator.lift(self.menu)

        elif self.zoneShape.get()=='polygon':
            self.r_label.lift(self.menu)
            self.r_slider.lift(self.menu)
            self.r_indicator.lift(self.menu)
            
            self.sides_label.lift(self.menu)
            self.sides_slider.lift(self.menu)
            self.sides_indicator.lift(self.menu)
            
            self.angle_label.lift(self.menu)
            self.angle_slider.lift(self.menu)
            self.angle_indicator.lift(self.menu)

        #Redrawing with new shape
        self.zoneDraw()
            
        



    #Resizing the image to match frame size
    def resize(self, event):
        if self.PILimage is not None:
            size = (event.width, event.height)
            self.displayCVImage(self.dispIm, size)
        else:
            pass


    #Displays the image in the frame
    def displayCVImage(self, im, size=None):
        self.dispIm = im
        if size is None:
            size = (self.frame.winfo_width(), self.frame.winfo_height())

        size = (size[0]-sizeFudge, size[1]-sizeFudge)
        self.PILimage = Image.fromarray(cv2.cvtColor(cv2.resize(im, size), cv2.COLOR_BGR2RGB)) #opencv stores images in bgr, PIL in rgb
        self.Tkimage = ImageTk.PhotoImage(self.PILimage)
        self.display.config(image = self.Tkimage)
        self.display.image = self.Tkimage        


    #Plots the histogram
    def plotHistogram(self):

        #Getting the boolean options for what to plot
        plotBlue = self.plotHistoBlue.get()
        plotGreen = self.plotHistoGreen.get()
        plotRed = self.plotHistoRed.get()

        #Only computing histograms for the channels that are requested
        if plotBlue:
            Bheights, edges = imageChannelHistogram(self.im[:,:,0], self.mask)
        else:
            Bheights = self.blankHeights
        if plotGreen:
            Gheights, edges = imageChannelHistogram(self.im[:,:,1], self.mask)
        else:
            Gheights = self.blankHeights
        if plotRed:
            Rheights, edges = imageChannelHistogram(self.im[:,:,2], self.mask)
        else:
            Rheights = self.blankHeights

        plotIntersections = self.plotHistoIntersection.get()

        #The only way to update a matplotlib bar chart without making a whole new one
        #   is to loop through each bar and modify it
        for i in range(len(self.centers)):

            #Setting the heights for the RGB plots
            self.histoPlotBlue[i].set_height(Bheights[i])
            self.histoPlotGreen[i].set_height(Gheights[i])
            self.histoPlotRed[i].set_height(Rheights[i])

            #If the user has elected to show the intersections, show them
            if plotIntersections:
                self.histoPlotCyan[i].set_height(np.min([Bheights[i],Gheights[i]]))
                self.histoPlotMagenta[i].set_height(np.min([Bheights[i],Rheights[i]]))
                self.histoPlotYellow[i].set_height(np.min([Rheights[i],Gheights[i]]))

                self.histoPlotWhite[i].set_height(np.min([Bheights[i],Gheights[i],Rheights[i]]))

            #Otherwise, give them blank histograms
            else:
                self.histoPlotCyan[i].set_height(self.blankHeights[i])
                self.histoPlotMagenta[i].set_height(self.blankHeights[i])
                self.histoPlotYellow[i].set_height(self.blankHeights[i])

                self.histoPlotWhite[i].set_height(self.blankHeights[i])



        #Rescale to match new max height
        self.histoAxis.set_ylim([0,np.max([Bheights, Gheights, Rheights])])

        #Redraw the figure
        self.histoFig.canvas.draw()
        self.histoFig.canvas.flush_events()

        #This pause is required to keep it from breaking
        plt.pause(0.000000000001)
        

    #Displays the zone on the image
    def zoneDraw(self, e=None):

        #Copies the image
        self.imDraw = self.im.copy()

        #Restricting the shape parameters to integer values
        self.displace_x.set(int(self.x_slider.get()))
        self.displace_y.set(int(self.y_slider.get()))
        self.radius.set(int(self.r_slider.get()))
        self.rectangle_width.set(int(self.width_slider.get()))
        self.rectangle_height.set(int(self.height_slider.get()))
        self.nsides.set(int(self.sides_slider.get()))

        #Making sure the angle slider is between its min/max values (will change with # sides)
        self.angle_slider.configure(to=180/self.nsides.get(), from_=-180/self.nsides.get())
        if np.abs(self.poly_angle.get())>180/self.nsides.get():
            self.angle_slider.set(np.sign(self.angle_slider.get())*180/self.nsides.get())

        #Converting the angle in degrees to angle in radians
        poly_angle_rad = np.deg2rad(self.poly_angle.get())

        shape = self.zoneShape.get()

        #Shifting the center of the shape by the displacement
        centershift = self.center+np.array([self.displace_x.get(), -self.displace_y.get()])

        #Initializing a mask to be drawn on
        self.mask = np.zeros(self.imDraw.shape[:2], dtype=np.uint8)

        #Encapsulating the information into a shape data array
        if shape=='polygon':
            self.data = [self.nsides.get(), poly_angle_rad, [self.radius.get()]]
        elif shape=='circle':
            self.data = [[self.radius.get()]]
        elif shape=='rectangle':
            self.data = [[self.rectangle_width.get(), self.rectangle_height.get()]]    
                                  
        #Draws the shape onto the mask
        drawShape(self.mask, self.zoneShape.get(), centershift, self.data, color=255, thickness=-1)

        #Computing the area by summing over the mask
        self.area.set(np.sum(self.mask)//255)

        #If we aren't looking at the masked zone, draw on the image
        if not self.maskZone.get():
            drawShape(self.imDraw, self.zoneShape.get(), centershift, self.data, color=(128,0,128), thickness=1)

        #Otherwise, mask the image with the shape
        else:
            self.imDraw = cv2.bitwise_and(self.im, self.im, mask=self.mask)

        #If the user has elected to show the histogram
        if self.showHist.get():
            self.plotHistogram()

        #Update the displayed image
        self.displayCVImage(self.imDraw)

    #Returns the refiner parameters
    def getParams(self):
        return self.zoneShape.get(), self.displace_x.get(), self.displace_y.get(), self.data

        

#Object: ColorGUI
#Purpose: Main application window, displays image
class ColorGUI:

    def __init__(self, window):


        self.window = window
        window.title("ColorScan")

        #Frames for the menu and the image
        self.menu = ttk.Frame(self.window)
        self.frame = ttk.Frame(self.window, borderwidth=10, relief="sunken", width=600, height=600)
        
        self.menu.grid(column=0, row=0, sticky='nesw')
        self.frame.grid(column=1, row=0, columnspan=1, rowspan=1, sticky='nesw')


        #Making it so that the image frame resizes with the window
        self.window.grid_rowconfigure(0, weight=1)
        self.window.grid_columnconfigure(1, weight=1)



        #Initializing variables
        self.filePath = ""
        self.image = None
        self.dispIm = self.image
        self.PILimage = None
        self.Tkimage = None

        #Starting at row 0
        row = 0

        #Button to select the image
        self.imSelectButton = ttk.Button(self.menu, text="Select Image", command = self.getImg)
        self.imSelectButton.grid(row=row, column=0, sticky='we')

        row+=1
        ###Next Row###

        #Button to start analysis
        self.analysisButton = ttk.Button(self.menu, text="Analysis", command = self.analyze)
        self.analysisButton.grid(row=row, column=0, sticky='we')


        row+=1
        ###Next Row###
        
        spacer = ttk.Label(master=self.menu, text="")
        spacer.grid(row=row)

        self.menu.grid_rowconfigure(row, weight=1)
        
        row+=1
        ###Next Row###

        #Checkbutton to maintain the original aspect ratio of the image when resizing
        self.fixAspect = tk.BooleanVar(value=True)
        self.aspectCheck = ttk.Checkbutton(self.menu, text="Fix Aspect Ratio", variable=self.fixAspect, onvalue=True, command=self.displayCVImage)
        self.aspectCheck.grid(row=row, column=0, sticky='we')


        #When analysis pane is None it indicates that the user has not started analysis yet
        self.analysisPane = None

        #Setting up the display for the image
        self.display = ttk.Label(self.frame, text="No Image Selected")
        self.display.grid(row=0, column=0)

        self.frame.grid_columnconfigure(0, weight=1)
        self.frame.grid_rowconfigure(0, weight=1)


        #Making it so that resizing the frame resizes the image
        self.frame.bind("<Configure>", self.resize)

        #Press ESC to quit
        self.window.bind("<Key>", lambda event: self.window.quit() if event.keycode==ESC else 0)

        #Allows the user to save a snapshot of the analysis image at any time
        self.snapshots = tk.IntVar(master=self.window, value=0)
        self.window.bind("s", lambda event: \
            [cv2.imwrite(os.path.splitext(self.filePath)[0]+"_snapshot_"+\
                         str(self.snapshots.get())+self.ext,self.dispIm),\
             self.snapshots.set(self.snapshots.get()+1), print("saved snapshot")])

        #Prevents the resizing of the image from resizing the window (preventing feedback loops)
        self.window.grid_propagate(False)

    #Resizes the image to match frame size
    def resize(self, event):
        #If the user has selected an image
        if self.PILimage is not None:
            #If the user has elected to maintain the aspect ratio
            if self.fixAspect.get():                                           
                size = windowAspectAdjust((event.width,event.height), self.dispIm, scaling=1)
            #Otherwise the image size will be that of the frame
            else:
                size = (event.width, event.height)

            self.displayCVImage(self.dispIm, size)
        else:
            pass


    #Prompts the user to select an image, then reads it
    def getImg(self):

        #Opens a dialog box to find an image
        filePath = tk.filedialog.askopenfilename()

        #If the user successfully picked a file
        if filePath!="":
            #If the user has already started an analysis, close the pane
            #   before opening a new image
            if self.analysisPane is not None:
                self.analysisPane.close()

            #Extracting the path, filename, and extension
            self.filePath=filePath
            self.ext = os.path.splitext(self.filePath)[-1]
            self.filename = os.path.splitext(self.filePath.split('/')[-1])[0]
            
            print("Selected image:", self.filePath)

            #Will fail if the user has selected something that isn't an image
            try:
                self.image = cv2.imread(self.filePath)
                self.dispIm = self.image

                #Resize the image to be about 2/3 the size of the screen while keeping
                #   the aspect ratio constant
                screenSize = (self.window.winfo_screenwidth(), self.window.winfo_screenheight())
                width, height = windowAspectAdjust(screenSize, self.image, scaling=2/3)
                self.displayCVImage(self.image, (width, height))

                #resizes the window and moves it to the top left corner
                self.window.geometry(str(width)+'x'+str(height)+"+0+0")
                
            #When it fails, alert the user
            except AttributeError:
                print("Bad file type! Pick a different image.")
            
    #Displays an image in the frame
    def displayCVImage(self, im=None, size=None):
        if im is not None:        
            self.dispIm = im
        else:
            im = self.dispIm
        if size is None:
            size = (self.frame.winfo_width(), self.frame.winfo_height())

        if self.fixAspect.get():                                           
            size = windowAspectAdjust(size, self.dispIm, scaling=1)

        size = (size[0]-sizeFudge, size[1]-sizeFudge)
        self.PILimage = Image.fromarray(cv2.cvtColor(cv2.resize(im, size), cv2.COLOR_BGR2RGB)) #opencv stores images in bgr, PIL in rgb
        self.Tkimage = ImageTk.PhotoImage(self.PILimage)
        self.display.config(image = self.Tkimage)
        self.display.image = self.Tkimage


    #Starts analysis
    def analyze(self):
        if self.image is None:
            print("No image selected!")
        else:
            if self.analysisPane is not None:
                self.analysisPane.close()

            self.analysisWindow = tk.Toplevel(master=self.window)
            self.analysisPane = AnalysisWindow(self.analysisWindow, self)

if __name__=='__main__':
    print("Done loading, welcome to ColorScan")
    #gui instance of the ColorGUI class
    gui = ColorGUI(root)

    #Starts the application -- pauses here while running
    root.mainloop()

    #Sometimes pressing escape doesn't close the window, this catches it
    try:
        root.destroy()
    except tk.TclError:
        pass


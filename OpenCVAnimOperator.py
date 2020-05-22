import bpy
import cv2
import time
import numpy

# Download trained model (lbfmodel.yaml)
# https://github.com/kurnianggoro/GSOC2017/tree/master/data

# Install prerequisites:

# Linux: (may vary between distro's and installation methods)
# This is for manjaro with Blender installed from the package manager
# python3 -m ensurepip
# python3 -m pip install --upgrade pip --user
# python3 -m pip install opencv-contrib-python numpy --user

# MacOS
# open the Terminal
# cd /Applications/Blender.app/Contents/Resources/2.81/python/bin
# ./python3.7m -m ensurepip
# ./python3.7m -m pip install --upgrade pip --user
# ./python3.7m -m pip install opencv-contrib-python numpy --user

# Windows:
# Open Command Prompt as Administrator
# cd "C:\Program Files\Blender Foundation\Blender 2.81\2.81\python\bin"
# python -m pip install --upgrade pip
# python -m pip install opencv-contrib-python numpy
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale              = 0.5
fontColor              = (255,255,255)
lineType               = 2
def drawMarkerNumsOnFrame(image,faceShape):
    '''
        draw colored markers for ease of understanding,
        red: head orientation 6 points
        green: mouth width
        blue: mouth height
        yellow: brows
        purple: jaw game
        cyan: nose

    '''

    global font, fontScale,fontColor,lineType
    
    
    # USING MARKER NUMBERS ON SCREEL APPROACH 1
    markers = [38,40,43,47] # [nose,chin,eye_L,eye_R, mouth_L, mouth_R]
    markersPos = [faceShape[38],faceShape[40], faceShape[43], faceShape[47]]
    i = 0
    for marker in markers:
        cv2.putText(image,str(marker), (markersPos[i][0],markersPos[i][1]), font, fontScale, fontColor, lineType)
        i = i+1

    # USE BETTER COLORING 
    headPos = [faceShape[30],faceShape[8], faceShape[36], faceShape[45], faceShape[48], faceShape[54]]
    for each in headPos:
        cv2.circle(image, (each[0],each[1]), 4, (255, 0, 0), -1)
    mwPos = [faceShape[48],faceShape[54]]
    for each in mwPos:
        cv2.circle(image, (each[0],each[1]), 3, (0, 255, 0), -1)
    mhPos = [faceShape[62],faceShape[66]]
    for each in mhPos:
        cv2.circle(image, (each[0],each[1]), 2, (0, 0, 255), -1)


class OpenCVAnimOperator(bpy.types.Operator):
    """Operator which runs its self from a timer"""
    bl_idname = "wm.opencv_operator"
    bl_label = "OpenCV Animation Operator"

    # Set paths to trained models downloaded above
    face_detect_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    landmark_model_path = "/home/user/Desktop/bdev/MC4B/models/lbfmodel.yaml"  #Linux
    #landmark_model_path = "/Users/username/Downloads/lbfmodel.yaml"         #Mac
    #landmark_model_path = "C:\\Users\\me\\Desktop\\cvmc2\\data\\lbfmodel.yaml"    #Windows

    # Load models
    fm = cv2.face.createFacemarkLBF()
    fm.loadModel(landmark_model_path)
    cas = cv2.CascadeClassifier(face_detect_path)

    _timer = None
    _cap  = None
    stop = False

    # Webcam resolution:
    width = 640
    height = 480

    # Choose a font
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,400)
    fontScale              = 1
    fontColor              = (255,255,255)
    lineType               = 2

    # 3D model points.
    model_points = numpy.array([
                                (0.0, 0.0, 0.0),             # Nose tip
                                (0.0, -330.0, -65.0),        # Chin
                                (-225.0, 170.0, -135.0),     # Left eye left corner
                                (225.0, 170.0, -135.0),      # Right eye right corne
                                (-150.0, -150.0, -125.0),    # Left Mouth corner
                                (150.0, -150.0, -125.0)      # Right mouth corner
                            ], dtype = numpy.float32)
    # Camera internals
    camera_matrix = numpy.array(
                            [[height, 0.0, width/2],
                            [0.0, height, height/2],
                            [0.0, 0.0, 1.0]], dtype = numpy.float32
                            )

    # Keeps a moving average of given length
    def smooth_value(self, name, length, value):
        if not hasattr(self, 'smooth'):
            self.smooth = {}
        if not name in self.smooth:
            self.smooth[name] = numpy.array([value])
        else:
            self.smooth[name] = numpy.insert(arr=self.smooth[name], obj=0, values=value)
            if self.smooth[name].size > length:
                self.smooth[name] = numpy.delete(self.smooth[name], self.smooth[name].size-1, 0)
        sum = 0
        for val in self.smooth[name]:
            sum += val
        return sum / self.smooth[name].size

    # Keeps min and max values, then returns the value in a range 0 - 1
    def get_range(self, name, value):
        if not hasattr(self, 'range'):
            self.range = {}
        if not name in self.range:
            self.range[name] = numpy.array([value, value])
        else:
            self.range[name] = numpy.array([min(value, self.range[name][0]), max(value, self.range[name][1])] )
        val_range = self.range[name][1] - self.range[name][0]
        if val_range != 0:
            return (value - self.range[name][0]) / val_range
        else:
            return 0.0

    # The main "loop"
    def modal(self, context, event):

        if (event.type in {'RIGHTMOUSE', 'ESC'}) or self.stop == True:
            self.cancel(context)
            return {'CANCELLED'}

        if event.type == 'TIMER':
            self.init_camera()
            _, image = self._cap.read()
            #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            #gray = cv2.equalizeHist(gray)

            # find faces
            faces = self.cas.detectMultiScale(image,
                scaleFactor=1.05,
                minNeighbors=3,
                flags=cv2.CASCADE_SCALE_IMAGE,
                minSize=(int(self.width/5), int(self.width/5)))

            #find biggest face, and only keep it
            if type(faces) is numpy.ndarray and faces.size > 0:
                biggestFace = numpy.zeros(shape=(1,4))
                for face in faces:
                    if face[2] > biggestFace[0][2]:
                        # print(face)
                        biggestFace[0] = face

                # find the landmarks.
                _, landmarks = self.fm.fit(image, faces=biggestFace)
                for mark in landmarks:
                    shape = mark[0]
                    # print ('NOSE:',shape[30])

                    #2D image points. If you change the image, you need to change vector
                    image_points = numpy.array([shape[30],     # Nose tip - 31
                                                shape[8],      # Chin - 9
                                                shape[36],     # Left eye left corner - 37
                                                shape[45],     # Right eye right corne - 46
                                                shape[48],     # Left Mouth corner - 49
                                                shape[54]      # Right mouth corner - 55
                                            ], dtype = numpy.float32)

                    dist_coeffs = numpy.zeros((4,1)) # Assuming no lens distortion
                    print ('IMAGE_POINTS:',image_points)
                    # determine head rotation
                    if hasattr(self, 'rotation_vector'):
                        (success, self.rotation_vector, self.translation_vector) = cv2.solvePnP(self.model_points,
                            image_points, self.camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE,
                            rvec=self.rotation_vector, tvec=self.translation_vector,
                            useExtrinsicGuess=True)
                    else:
                        (success, self.rotation_vector, self.translation_vector) = cv2.solvePnP(self.model_points,
                            image_points, self.camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE,
                            useExtrinsicGuess=False)

                    if not hasattr(self, 'first_angle'):
                        self.first_angle = numpy.copy(self.rotation_vector)

                    # set bone rotation/positions
                    bones = bpy.data.objects["rig"].pose.bones

                    # head rotation
                    #bones["head"].rotation_euler[0] = self.smooth_value("h_x", 5, (self.rotation_vector[0] - self.first_angle[0])) / 1   # Up/Down
                    #bones["head"].rotation_euler[2] = self.smooth_value("h_y", 5, -(self.rotation_vector[1] - self.first_angle[1])) / 1.5  # Rotate
                    #bones["head"].rotation_euler[1] = self.smooth_value("h_z", 5, (self.rotation_vector[2] - self.first_angle[2])) / 1.3   # Left/Right

                    #bones["head"].keyframe_insert(data_path="rotation_euler", index=-1)

                    # mouth position
                    #bones["jaw_master"].location[2] = self.smooth_value("m_h", 2, -self.get_range("mouth_height", numpy.linalg.norm(shape[62] - shape[66])) * 0.06 )
                    #bones["jaw_master"].location[0] = self.smooth_value("m_w", 2, (self.get_range("mouth_width", numpy.linalg.norm(shape[54] - shape[48])) - 0.5) * 0.04)
                    #bones["jaw_master"].keyframe_insert(data_path="location", index=-1)

                    #eyebrows inner side
                    bones["brow.T.L.003"].location[1] = self.smooth_value("b_t_l_i", 3, (self.get_range("brow_left_i", numpy.linalg.norm(shape[21] - shape[27])) -0.5) * 0.04)
                    bones["brow.T.R.003"].location[1] = self.smooth_value("b_t_r_i", 3, (self.get_range("brow_right_i", numpy.linalg.norm(shape[22] - shape[27])) -0.5) * 0.04)
                    bones["brow.T.L.003"].keyframe_insert(data_path="location", index=1)
                    bones["brow.T.R.003"].keyframe_insert(data_path="location", index=1)
                    #eyebrows outer side
                    bones["brow.T.L.001"].location[1] = self.smooth_value("b_t_l_o", 3, (self.get_range("brow_left_o", numpy.linalg.norm(shape[17] - shape[27])) -0.5) * 0.04)
                    bones["brow.T.R.001"].location[1] = self.smooth_value("b_t_r_o", 3, (self.get_range("brow_right_o", numpy.linalg.norm(shape[26] - shape[27])) -0.5) * 0.04)
                    bones["brow.T.L.001"].keyframe_insert(data_path="location", index=1)
                    bones["brow.T.R.001"].keyframe_insert(data_path="location", index=1)
                                        

                    # eyelids
                    l_open = self.smooth_value("e_l", 2, self.get_range("l_open", -numpy.linalg.norm(shape[38] - shape[40]))  )
                    r_open = self.smooth_value("e_r", 2, self.get_range("r_open", -numpy.linalg.norm(shape[43] - shape[47]))  )
                    eyes_open = (l_open + r_open) / 2.0 # looks weird if both eyes aren't the same...
                    bones["lid.T.R.002"].location[1] =   -eyes_open * 0.025 + 0.005
                    bones["lid.B.R.002"].location[1] =  eyes_open * 0.025 - 0.005
                    bones["lid.T.L.002"].location[1] =   -eyes_open * 0.025 + 0.005
                    bones["lid.B.L.002"].location[1] =  eyes_open * 0.025 - 0.005

                    bones["lid.T.R.002"].keyframe_insert(data_path="location", index=1)
                    bones["lid.B.R.002"].keyframe_insert(data_path="location", index=1)
                    bones["lid.T.L.002"].keyframe_insert(data_path="location", index=1)
                    bones["lid.B.L.002"].keyframe_insert(data_path="location", index=1)

                    # draw face markers
                    for (x, y) in shape:
                        cv2.circle(image, (x, y), 2, (0, 255, 255), -1)
                    # DRAW SPECIAL
                    # TODO make lower function call optional from gui ( for debugging purposes )
                    drawMarkerNumsOnFrame(image,shape)


            # draw detected face
            for (x,y,w,h) in faces:
                cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),1)
            
            #cv2.putText(image,'Hello World!', self.bottomLeftCornerOfText, self.font, self.fontScale, self.fontColor, self.lineType)
            # Show camera image in a window
            cv2.imshow("Output", image)
            cv2.waitKey(1)

        return {'PASS_THROUGH'}

    def init_camera(self):
        if self._cap == None:
            self._cap = cv2.VideoCapture(0)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            time.sleep(1.0)

    def stop_playback(self, scene):
        print(format(scene.frame_current) + " / " + format(scene.frame_end))
        if scene.frame_current == scene.frame_end:
            bpy.ops.screen.animation_cancel(restore_frame=False)

    def execute(self, context):
        bpy.app.handlers.frame_change_pre.append(self.stop_playback)

        wm = context.window_manager
        self._timer = wm.event_timer_add(0.01, window=context.window)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def cancel(self, context):
        wm = context.window_manager
        wm.event_timer_remove(self._timer)
        cv2.destroyAllWindows()
        self._cap.release()
        self._cap = None

def register():
    bpy.utils.register_class(OpenCVAnimOperator)

def unregister():
    bpy.utils.unregister_class(OpenCVAnimOperator)

if __name__ == "__main__":
    register()

    # test call
    #bpy.ops.wm.opencv_operator()

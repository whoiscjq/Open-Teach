import numpy as np
from copy import deepcopy as copy
from openteach.components import Component
from openteach.constants import *
from openteach.utils.vectorops import *
from openteach.utils.network import ZMQKeypointPublisher, ZMQKeypointSubscriber
from openteach.utils.timer import FrequencyTimer

# This class is used to transform the left hand coordinates from the VR to the robot frame.
class TransformHandPositionCoords(Component):
    def __init__(self, host, keypoint_port, transformation_port, moving_average_limit = 10):
        self.notify_component_start('keypoint position transform')
        
        # Initializing the subscriber for left hand keypoints
        self.original_keypoint_subscriber_left = ZMQKeypointSubscriber(host, keypoint_port, 'left')
        self.original_keypoint_subscriber_right = ZMQKeypointSubscriber(host, keypoint_port, 'right')
        # Initializing the publisher for transformed left hand keypoints
        self.transformed_keypoint_publisher = ZMQKeypointPublisher(host, transformation_port)

        self.timer = FrequencyTimer(VR_FREQ)
        
        # Keypoint indices for knuckles
        self.knuckle_points = (OCULUS_JOINTS['knuckles'][0], OCULUS_JOINTS['knuckles'][-1])

        # Moving average queue
        self.moving_average_limit = moving_average_limit
        self.coord_moving_average_queue, self.frame_moving_average_queue = [], []

    def _get_hand_coords_left(self):

        # This is for getting hand keypoints from VR.
        data = self.original_keypoint_subscriber_left.recv_keypoints()
        if data[0] == 0:
            data_type = 'absolute'
        else:
            data_type = 'relative'

        return data_type, np.asanyarray(data[1:]).reshape(OCULUS_NUM_KEYPOINTS, 3)


    def _get_hand_coords_right(self):

        # This is for getting hand keypoints from VR.
        data = self.original_keypoint_subscriber_right.recv_keypoints()
        if data[0] == 0:
            data_type = 'absolute'
        else:
            data_type = 'relative'

        return data_type, np.asanyarray(data[1:]).reshape(OCULUS_NUM_KEYPOINTS, 3)

    def _translate_coords(self, hand_coords):
        #Find the vectors with respect to the wrist.
        return copy(hand_coords) - hand_coords[0]

    # Create a coordinate frame for the hand
    def _get_coord_frame(self, index_knuckle_coord, pinky_knuckle_coord):

        #This function is used for retargeting hand keypoints from oculus. The frames here are in robot frame.
        palm_normal = normalize_vector(np.cross(index_knuckle_coord, pinky_knuckle_coord))   # Current Z
        palm_direction = normalize_vector(index_knuckle_coord + pinky_knuckle_coord)         # Current Y
        cross_product =normalize_vector(np.cross(palm_direction, palm_normal))              # Current X
        
        # palm_normal = np.cross(index_knuckle_coord, pinky_knuckle_coord)   # Current Z
        # palm_direction = index_knuckle_coord + pinky_knuckle_coord        # Current Y
        # cross_product = np.cross(palm_direction, palm_normal)    
        
        return [cross_product, palm_direction, palm_normal]

    # Create a coordinate frame for the arm
    def _get_hand_dir_frame(self, origin_coord, index_knuckle_coord, pinky_knuckle_coord):
        # Calculating the transform in Left Handed system itself as unity being left hand coordinate system. This transform sends the frame in the form of vectors in Left Hand coordinate frame. Since we use only the relative transform between the hand movements the coordinate system does not matter. 
        # We find the relative transformation between the hand moving frames and use that to find the transformation in the robot frame and this does not depend on the coordinate system
        
        palm_normal = normalize_vector(np.cross(pinky_knuckle_coord,index_knuckle_coord))   # Unity space Y  
        palm_direction = normalize_vector(pinky_knuckle_coord+index_knuckle_coord)         # Unity space Z            
        cross_product = normalize_vector(pinky_knuckle_coord-index_knuckle_coord)         # Unity space X 

        return [origin_coord, cross_product, palm_normal, palm_direction]

    # Create a coordinate frame for the arm
    def _get_hand_dir_frame_left(self, origin_coord, index_knuckle_coord, pinky_knuckle_coord):
        # Calculating the transform in Left Handed system itself as unity being left hand coordinate system. This transform sends the frame in the form of vectors in Left Hand coordinate frame. Since we use only the relative transform between the hand movements the coordinate system does not matter. 
        # We find the relative transformation between the hand moving frames and use that to find the transformation in the robot frame and this does not depend on the coordinate system
        
        palm_normal = normalize_vector(np.cross(pinky_knuckle_coord,index_knuckle_coord))   # Unity space Y  
        palm_direction = normalize_vector(pinky_knuckle_coord+index_knuckle_coord)         # Unity space Z            
        cross_product = normalize_vector(pinky_knuckle_coord-index_knuckle_coord)         # Unity space X 

        return [origin_coord, cross_product, palm_normal, palm_direction]



    # Function to transform the hand keypoints to the robot frame
    def transform_keypoints_left(self, hand_coords):
       
        translated_coords = self._translate_coords(hand_coords)  # Finding hand coords with respect to the wrist

        original_coord_frame = self._get_coord_frame(           # Finding the original coordinate frame
            translated_coords[self.knuckle_points[0]], 
            translated_coords[self.knuckle_points[1]]
        )
       
        rotation_matrix = np.linalg.solve(original_coord_frame, np.eye(3)).T # Finding the rotation matrix and rotating the coordinates
        
        transformed_hand_coords = (rotation_matrix @ translated_coords.T).T # Find the transformed hand coordinates in the robot frame
        # Find the transformed arm frame using knuckle coords
        hand_dir_frame = self._get_hand_dir_frame_left(
            hand_coords[0],
            translated_coords[self.knuckle_points[0]], 
            translated_coords[self.knuckle_points[1]]
        )

        return transformed_hand_coords, hand_dir_frame
    
    def transform_keypoints_right(self, hand_coords):
    
        translated_coords = self._translate_coords(hand_coords)  # Finding hand coords with respect to the wrist

        original_coord_frame = self._get_coord_frame(           # Finding the original coordinate frame
            translated_coords[self.knuckle_points[0]], 
            translated_coords[self.knuckle_points[1]]
        )
        
        rotation_matrix = np.linalg.solve(original_coord_frame, np.eye(3)).T # Finding the rotation matrix and rotating the coordinates
        
        transformed_hand_coords = (rotation_matrix @ translated_coords.T).T # Find the transformed hand coordinates in the robot frame
        # Find the transformed arm frame using knuckle coords
        hand_dir_frame = self._get_hand_dir_frame(
            hand_coords[0],
            translated_coords[self.knuckle_points[0]], 
            translated_coords[self.knuckle_points[1]]
        )

        return transformed_hand_coords, hand_dir_frame
    
    

    def stream(self):
        while True:
            try:
                self.timer.start_loop()
                # Get the hand coordinates
                data_type, hand_coords_left = self._get_hand_coords_left()
                data_type, hand_coords_right = self._get_hand_coords_right()

                # Find the transformed hand coordinates and the transformed hand local frame
                transformed_hand_coords_left, translated_hand_coord_frame_left = self.transform_keypoints_left(hand_coords_left)
                transformed_hand_coords_right, translated_hand_coord_frame_right = self.transform_keypoints_right(hand_coords_right)
 
                
                # transformed_hand_coords=np.stack((transformed_hand_coords_right[0], transformed_hand_coords_left[1:]),axis=0)
                #translated_hand_coord_frame=np.stack((translated_hand_coord_frame_right[0], translated_hand_coord_frame_left[1:]),axis=0)
                # print(transformed_hand_coords_right[0].shape)
                # print(transformed_hand_coords_right[1:].shape)
                #translated_hand_coord_frame=np.stack((translated_hand_coord_frame_right[0], translated_hand_coord_frame_left[1:]),axis=0)
                #print(translated_hand_coord_frame.shape)
                
                # Passing the transformed coords into a moving average to filter the noise. The higher the moving average limit, the more the noise is filtered. But values higher than 50 might make the keypoint less responsive.
                self.averaged_hand_coords = moving_average(
                    transformed_hand_coords_right, 
                    self.coord_moving_average_queue, 
                    self.moving_average_limit
                )

                # Passing the transformed frame into a moving average to filter the noise. The higher the moving average limit, the more the noise is filtered. But the
                averaged_hand_frame_left = moving_average(
                    translated_hand_coord_frame_left, 
                    self.frame_moving_average_queue, 
                    self.moving_average_limit
                )
                
                averaged_hand_frame_right = moving_average(
                    translated_hand_coord_frame_right, 
                    self.frame_moving_average_queue, 
                    self.moving_average_limit
                )
                # Publish the transformed hand coordinates
                self.transformed_keypoint_publisher.pub_keypoints(self.averaged_hand_coords, 'transformed_hand_coords')
                # Publish the transformed hand frame
                if data_type == 'absolute':
                    self.averaged_hand_frame=np.concatenate((averaged_hand_frame_right[0].reshape(1,-1), averaged_hand_frame_left[1:]),axis=0)
                    #self.averaged_hand_frame = (averaged_hand_frame_left + averaged_hand_frame_right)/2
                    #print(self.averaged_hand_fr
                    self.transformed_keypoint_publisher.pub_keypoints(self.averaged_hand_frame, 'transformed_hand_frame')
                    
                    
                
                # End the timer
                self.timer.end_loop()
            except:
                break
        # Stop the subscriber and publisher
        self.original_keypoint_subscriber_left.stop()
        self.original_keypoint_subscriber_right.stop()
        self.transformed_keypoint_publisher.stop()
        print('Stopping the keypoint position transform process.')
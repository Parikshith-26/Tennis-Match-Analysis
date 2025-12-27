import cv2
import numpy as np
import sys
sys.path.append('../')
import constants
from utils import (
    convert_meters_to_pixel_distance,
    convert_pixel_distance_to_meters,
    get_foot_position,
    get_closest_keypoint_index,
    get_height_of_bbox,
    measure_xy_distance,
    get_center_of_bbox,
    measure_distance
)

class MiniCourt():
    def __init__(self,frame):
        self.drawing_rectangle_width = 250
        self.drawing_rectangle_height = 500
        self.buffer = 50
        self.padding_court=20

        self.set_canvas_background_box_position(frame)
        self.set_mini_court_position()
        self.set_court_drawing_key_points()
        self.set_court_lines()


    def convert_meters_to_pixels(self, meters):
        return convert_meters_to_pixel_distance(meters,
                                                constants.DOUBLE_LINE_WIDTH,
                                                self.court_drawing_width
                                            )

    def set_court_drawing_key_points(self):
        drawing_key_points = [0]*28

        # point 0 
        drawing_key_points[0] , drawing_key_points[1] = int(self.court_start_x), int(self.court_start_y)
        # point 1
        drawing_key_points[2] , drawing_key_points[3] = int(self.court_end_x), int(self.court_start_y)
        # point 2
        drawing_key_points[4] = int(self.court_start_x)
        drawing_key_points[5] = self.court_start_y + self.convert_meters_to_pixels(constants.HALF_COURT_LINE_HEIGHT*2)
        # point 3
        drawing_key_points[6] = drawing_key_points[0] + self.court_drawing_width
        drawing_key_points[7] = drawing_key_points[5] 
        # #point 4
        drawing_key_points[8] = drawing_key_points[0] +  self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[9] = drawing_key_points[1] 
        # #point 5
        drawing_key_points[10] = drawing_key_points[4] + self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[11] = drawing_key_points[5] 
        # #point 6
        drawing_key_points[12] = drawing_key_points[2] - self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[13] = drawing_key_points[3] 
        # #point 7
        drawing_key_points[14] = drawing_key_points[6] - self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[15] = drawing_key_points[7] 
        # #point 8
        drawing_key_points[16] = drawing_key_points[8] 
        drawing_key_points[17] = drawing_key_points[9] + self.convert_meters_to_pixels(constants.NO_MANS_LAND_HEIGHT)
        # # #point 9
        drawing_key_points[18] = drawing_key_points[16] + self.convert_meters_to_pixels(constants.SINGLE_LINE_WIDTH)
        drawing_key_points[19] = drawing_key_points[17] 
        # #point 10
        drawing_key_points[20] = drawing_key_points[10] 
        drawing_key_points[21] = drawing_key_points[11] - self.convert_meters_to_pixels(constants.NO_MANS_LAND_HEIGHT)
        # # #point 11
        drawing_key_points[22] = drawing_key_points[20] +  self.convert_meters_to_pixels(constants.SINGLE_LINE_WIDTH)
        drawing_key_points[23] = drawing_key_points[21] 
        # # #point 12
        drawing_key_points[24] = int((drawing_key_points[16] + drawing_key_points[18])/2)
        drawing_key_points[25] = drawing_key_points[17] 
        # # #point 13
        drawing_key_points[26] = int((drawing_key_points[20] + drawing_key_points[22])/2)
        drawing_key_points[27] = drawing_key_points[21] 

        self.drawing_key_points=drawing_key_points

    def set_court_lines(self):
        self.lines = [
            (0, 2),
            (4, 5),
            (6,7),
            (1,3),
            
            (0,1),
            (8,9),
            (10,11),
            (10,11),
            (2,3)
        ]

    def set_mini_court_position(self):
        self.court_start_x = self.start_x + self.padding_court
        self.court_start_y = self.start_y + self.padding_court
        self.court_end_x = self.end_x - self.padding_court
        self.court_end_y = self.end_y - self.padding_court
        self.court_drawing_width = self.court_end_x - self.court_start_x

    def set_canvas_background_box_position(self,frame):
        frame= frame.copy()

        self.end_x = frame.shape[1] - self.buffer
        self.end_y = self.buffer + self.drawing_rectangle_height
        self.start_x = self.end_x - self.drawing_rectangle_width
        self.start_y = self.end_y - self.drawing_rectangle_height

    def draw_court(self,frame):
        for i in range(0, len(self.drawing_key_points),2):
            x = int(self.drawing_key_points[i])
            y = int(self.drawing_key_points[i+1])
            cv2.circle(frame, (x,y),5, (0,0,255),-1)

        # draw Lines
        for line in self.lines:
            start_point = (int(self.drawing_key_points[line[0]*2]), int(self.drawing_key_points[line[0]*2+1]))
            end_point = (int(self.drawing_key_points[line[1]*2]), int(self.drawing_key_points[line[1]*2+1]))
            cv2.line(frame, start_point, end_point, (0, 0, 0), 2)

        # Draw net
        net_start_point = (int(self.drawing_key_points[0]), int((self.drawing_key_points[1] + self.drawing_key_points[5])/2))
        net_end_point = (int(self.drawing_key_points[2]), int((self.drawing_key_points[1] + self.drawing_key_points[5])/2))
        cv2.line(frame, net_start_point, net_end_point, (255, 0, 0), 2)

        return frame

    def draw_background_rectangle(self,frame):
        shapes = np.zeros_like(frame,np.uint8)
        # Draw the rectangle
        cv2.rectangle(shapes, (self.start_x, self.start_y), (self.end_x, self.end_y), (255, 255, 255), cv2.FILLED)
        out = frame.copy()
        alpha=0.5
        mask = shapes.astype(bool)
        out[mask] = cv2.addWeighted(frame, alpha, shapes, 1 - alpha, 0)[mask]

        return out

    def draw_mini_court(self,frames):
        output_frames = []
        for frame in frames:
            frame = self.draw_background_rectangle(frame)
            frame = self.draw_court(frame)
            output_frames.append(frame)
        return output_frames

    def get_start_point_of_mini_court(self):
        return (self.court_start_x,self.court_start_y)
    def get_width_of_mini_court(self):
        return self.court_drawing_width
    def get_court_drawing_keypoints(self):
        return self.drawing_key_points

    def get_mini_court_coordinates(self,
                                   object_position,
                                   closest_key_point, 
                                   closest_key_point_index, 
                                   player_height_in_pixels,
                                   player_height_in_meters
                                   ):
        
        distance_from_keypoint_x_pixels, distance_from_keypoint_y_pixels = measure_xy_distance(object_position, closest_key_point)

        # Conver pixel distance to meters
        # protect against zero division by ensuring player_height_in_pixels > 0
        if player_height_in_pixels is None or player_height_in_pixels == 0:
            player_height_in_pixels = 1.0

        distance_from_keypoint_x_meters = convert_pixel_distance_to_meters(distance_from_keypoint_x_pixels,
                                                                           player_height_in_meters,
                                                                           player_height_in_pixels
                                                                           )
        distance_from_keypoint_y_meters = convert_pixel_distance_to_meters(distance_from_keypoint_y_pixels,
                                                                                player_height_in_meters,
                                                                                player_height_in_pixels
                                                                          )
        
        # Convert to mini court coordinates
        mini_court_x_distance_pixels = self.convert_meters_to_pixels(distance_from_keypoint_x_meters)
        mini_court_y_distance_pixels = self.convert_meters_to_pixels(distance_from_keypoint_y_meters)
        closest_mini_coourt_keypoint = ( int(self.drawing_key_points[closest_key_point_index*2]),
                                        int(self.drawing_key_points[closest_key_point_index*2+1])
                                        )
        
        mini_court_player_position = (closest_mini_coourt_keypoint[0]+mini_court_x_distance_pixels,
                                      closest_mini_coourt_keypoint[1]+mini_court_y_distance_pixels
                                        )

        return  mini_court_player_position

    def convert_bounding_boxes_to_mini_court_coordinates(self,player_boxes, ball_boxes, original_court_key_points ):
        """
        Robust conversion that:
         - handles missing frames
         - handles missing player/ball bboxes
         - returns two lists of length = max(len(player_boxes), len(ball_boxes))
           where each element is a dict: player -> (x,y) and ball -> {1: (x,y) or None}
        """
        player_heights = {
            1: constants.PLAYER_1_HEIGHT_METERS,
            2: constants.PLAYER_2_HEIGHT_METERS
        }

        max_len = max(len(player_boxes), len(ball_boxes))
        output_player_boxes= []
        output_ball_boxes= []

        for frame_num in range(max_len):
            # safe access to frame entries
            player_bbox = player_boxes[frame_num] if frame_num < len(player_boxes) else {}
            raw_ball_frame = ball_boxes[frame_num] if frame_num < len(ball_boxes) else {1: None}

            # get ball box safely
            ball_box = None
            if isinstance(raw_ball_frame, dict):
                ball_box = raw_ball_frame.get(1, None)
            elif isinstance(raw_ball_frame, list) and len(raw_ball_frame) > 0:
                # handle cases where stub stored [ [x1,y1,x2,y2] ] or similar
                first = raw_ball_frame[0]
                if isinstance(first, list) and len(first) == 4:
                    ball_box = first

            # compute ball center if available
            ball_position = None
            if ball_box is not None:
                try:
                    ball_position = get_center_of_bbox(ball_box)
                except Exception:
                    ball_position = None

            # prepare player outputs
            output_player_bboxes_dict = {}
            # collect heights to compute a reasonable player pixel-height fallback
            collected_heights = []

            # iterate players in this frame (may be empty)
            for player_id, bbox in player_bbox.items():
                if bbox is None:
                    continue

                # compute foot/anchor position
                try:
                    foot_position = get_foot_position(bbox)
                except Exception:
                    # fallback to bbox center if foot can't be computed
                    try:
                        foot_position = get_center_of_bbox(bbox)
                    except Exception:
                        continue

                # Get The closest keypoint in pixels
                try:
                    closest_key_point_index = get_closest_keypoint_index(foot_position, original_court_key_points, [0,2,12,13])
                except Exception:
                    closest_key_point_index = 0

                closest_key_point = (original_court_key_points[closest_key_point_index*2], 
                                     original_court_key_points[closest_key_point_index*2+1])

                # Get Player height in pixels (use surrounding frames safely)
                frame_index_min = max(0, frame_num-20)
                frame_index_max = min(max_len, frame_num+50)
                bboxes_heights_in_pixels = []
                for i in range(frame_index_min, frame_index_max):
                    if i < len(player_boxes) and isinstance(player_boxes[i], dict) and player_id in player_boxes[i]:
                        hb = player_boxes[i][player_id]
                        try:
                            h = get_height_of_bbox(hb)
                            if h is not None and h > 0:
                                bboxes_heights_in_pixels.append(h)
                        except Exception:
                            pass
                max_player_height_in_pixels = max(bboxes_heights_in_pixels) if bboxes_heights_in_pixels else None
                if max_player_height_in_pixels is None or max_player_height_in_pixels == 0:
                    # fallback to a sane default to avoid division by zero
                    max_player_height_in_pixels = 150.0

                collected_heights.append(max_player_height_in_pixels)

                # compute mini-court coordinates for this player
                mini_court_player_position = self.get_mini_court_coordinates(
                    foot_position,
                    closest_key_point, 
                    closest_key_point_index, 
                    max_player_height_in_pixels,
                    player_heights.get(player_id, constants.PLAYER_1_HEIGHT_METERS)
                )
                
                output_player_bboxes_dict[player_id] = mini_court_player_position

            # compute ball mini-court position:
            ball_mini_position = None
            if ball_position is not None:
                # choose a sensible pixel-height to convert from: use max of collected player heights or default
                fallback_player_pixel_height = max(collected_heights) if collected_heights else 150.0

                try:
                    ball_closest_key_point_index = get_closest_keypoint_index(ball_position, original_court_key_points, [0,2,12,13])
                except Exception:
                    ball_closest_key_point_index = 0

                ball_closest_key_point = (original_court_key_points[ball_closest_key_point_index*2],
                                          original_court_key_points[ball_closest_key_point_index*2+1])

                # convert ball center to mini-court coords using fallback height and an arbitrary player-meter height (use PLAYER_1)
                try:
                    ball_mini_position = self.get_mini_court_coordinates(
                        ball_position,
                        ball_closest_key_point,
                        ball_closest_key_point_index,
                        fallback_player_pixel_height,
                        constants.PLAYER_1_HEIGHT_METERS
                    )
                except Exception:
                    ball_mini_position = None

            # append outputs for this frame (ensure consistent length)
            output_player_boxes.append(output_player_bboxes_dict)
            output_ball_boxes.append({1: ball_mini_position} if ball_mini_position is not None else {1: None})

        return output_player_boxes , output_ball_boxes
    
    def draw_points_on_mini_court(self,frames,postions, color=(0,255,0)):
        """
        Draw points from 'positions' onto frames.
        'positions' is expected to be a list (per-frame) of dicts -> id:(x,y) or id:None
        This function safely skips None entries.
        """
        for frame_num, frame in enumerate(frames):
            if frame_num >= len(postions):
                continue
            frame_positions = postions[frame_num]
            if not isinstance(frame_positions, dict):
                continue
            for _, position in frame_positions.items():
                if position is None:
                    continue
                try:
                    x,y = position
                except Exception:
                    continue
                x= int(x)
                y= int(y)
                cv2.circle(frame, (x,y), 5, color, -1)
        return frames

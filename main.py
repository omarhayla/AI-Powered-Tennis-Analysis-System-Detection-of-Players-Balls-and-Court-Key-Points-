import numpy as np
import cv2
import pandas as pd
from copy import deepcopy
from utils import (read_video, save_video, measure_distance, draw_player_stats, convert_pixel_distance_to_meters)
import constants
from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector
from mini_court import MiniCourt


def main():
    # Read Video
    input_video_path = "input_videos/3.mp4"
    video_frames = read_video(input_video_path)

    # Detect Players and Ball
    player_tracker = PlayerTracker(model_path='models/yolo8trained.pt')
    ball_tracker = BallTracker(model_path='models/yolo5_last.pt')

    player_detections = player_tracker.detect_frames(video_frames,
                                                     read_from_stub=False,
                                                     stub_path="tracker_stubs/player_detections.pkl")
    ball_detections = ball_tracker.detect_frames(video_frames,
                                                 read_from_stub=False,
                                                 stub_path="tracker_stubs/ball_detections.pkl")
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)

    # Court Line Detector model
    court_model_path = "models/keypoints_model.pth"
    court_line_detector = CourtLineDetector(court_model_path)
    court_keypoints = court_line_detector.predict(video_frames[0])

    # Choose players
    player_detections = player_tracker.choose_and_filter_players(court_keypoints, player_detections)

    # MiniCourt
    mini_court = MiniCourt(video_frames[0])

    # Detect ball shots
    ball_shot_frames = ball_tracker.get_ball_shot_frames(ball_detections)

    # Convert positions to mini court positions
    player_mini_court_detections, ball_mini_court_detections = mini_court.convert_bounding_boxes_to_mini_court_coordinates(
        player_detections, ball_detections, court_keypoints)

    player_stats_data = [{
        'frame_num': 0,
        'player_1_number_of_shots': 0,
        'player_1_total_shot_speed': 0,
        'player_1_last_shot_speed': 0,
        'player_1_total_player_speed': 0,
        'player_1_last_player_speed': 0,
        'player_1_total_distance': 0,
        'player_1_favorite_side': 'N/A',
        'player_1_stamina': 100,
        'player_2_number_of_shots': 0,
        'player_2_total_shot_speed': 0,
        'player_2_last_shot_speed': 0,
        'player_2_total_player_speed': 0,
        'player_2_last_player_speed': 0,
        'player_2_total_distance': 0,
        'player_2_favorite_side': 'N/A',
        'player_2_stamina': 100,
        'player_side_time': {1: {'left': 0, 'right': 0}, 2: {'left': 0, 'right': 0}}
    }]

    for ball_shot_ind in range(len(ball_shot_frames) - 1):
        start_frame = ball_shot_frames[ball_shot_ind]
        end_frame = ball_shot_frames[ball_shot_ind + 1]
        ball_shot_time_in_seconds = (end_frame - start_frame) / 24  # 24fps

        distance_covered_by_ball_pixels = measure_distance(ball_mini_court_detections[start_frame][1],
                                                           ball_mini_court_detections[end_frame][1])
        distance_covered_by_ball_meters = convert_pixel_distance_to_meters(distance_covered_by_ball_pixels,
                                                                           constants.DOUBLE_LINE_WIDTH,
                                                                           mini_court.get_width_of_mini_court())

        # Speed of the ball shot in km/h
        speed_of_ball_shot = distance_covered_by_ball_meters / ball_shot_time_in_seconds * 3.6

        # Player who hit the ball
        player_positions = player_mini_court_detections[start_frame]
        player_shot_ball = min(player_positions.keys(),
                               key=lambda player_id: measure_distance(player_positions[player_id],
                                                                      ball_mini_court_detections[start_frame][1]))

        opponent_player_id = 1 if player_shot_ball == 2 else 2
        distance_covered_by_opponent_pixels = measure_distance(
            player_mini_court_detections[start_frame][opponent_player_id],
            player_mini_court_detections[end_frame][opponent_player_id])
        distance_covered_by_opponent_meters = convert_pixel_distance_to_meters(distance_covered_by_opponent_pixels,
                                                                               constants.DOUBLE_LINE_WIDTH,
                                                                               mini_court.get_width_of_mini_court())

        speed_of_opponent = distance_covered_by_opponent_meters / ball_shot_time_in_seconds * 3.6

        # Calculate cumulative distance covered by each player
        cumulative_distance_pixels = {1: 0, 2: 0}

        for i in range(start_frame, end_frame):
            for player_id in [1, 2]:
                player_box_frame1 = player_mini_court_detections[i][player_id]
                player_box_frame2 = player_mini_court_detections[i + 1][player_id]

                x1, y1 = player_box_frame1[0], player_box_frame1[1]
                x2, y2 = player_box_frame2[0], player_box_frame2[1]

                distance_pixels = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                cumulative_distance_pixels[player_id] += distance_pixels

        # Convert pixel distance to meters for each player
        distance_covered_meters = {player_id: convert_pixel_distance_to_meters(
            cumulative_distance_pixels[player_id],
            constants.DOUBLE_LINE_WIDTH,
            mini_court.get_width_of_mini_court()
        ) for player_id in [1, 2]}

        # Update stats
        current_player_stats = deepcopy(player_stats_data[-1])
        current_player_stats['frame_num'] = start_frame

        current_player_stats[f'player_{player_shot_ball}_number_of_shots'] += 1
        current_player_stats[f'player_{player_shot_ball}_total_shot_speed'] += speed_of_ball_shot
        current_player_stats[f'player_{player_shot_ball}_last_shot_speed'] = speed_of_ball_shot
        current_player_stats[f'player_{opponent_player_id}_total_player_speed'] += speed_of_opponent
        current_player_stats[f'player_{opponent_player_id}_last_player_speed'] = speed_of_opponent

        # Determine favorite side
        for player_id in [1, 2]:
            x_position = player_mini_court_detections[start_frame][player_id][0]
            court_midline = mini_court.get_width_of_mini_court() / 2
            side = 'left' if x_position < court_midline else 'right'
            current_player_stats['player_side_time'][player_id][side] += 1

            favorite_side = max(current_player_stats['player_side_time'][player_id],
                                key=current_player_stats['player_side_time'][player_id].get)
            current_player_stats[f'player_{player_id}_favorite_side'] = favorite_side.capitalize()

        for player_id in [1, 2]:
            current_player_stats[f'player_{player_id}_total_distance'] += distance_covered_meters[player_id]
            current_player_stats[f'player_{player_id}_stamina'] -= distance_covered_meters[player_id] * 0.2
            current_player_stats[f'player_{player_id}_stamina'] = max(0, min(100, current_player_stats[
                f'player_{player_id}_stamina']))

        player_stats_data.append(current_player_stats)

    # Convert stats to DataFrame
    player_stats_data_df = pd.DataFrame(player_stats_data)

    # Merge with frame numbers
    frames_df = pd.DataFrame({'frame_num': list(range(len(video_frames)))})
    player_stats_data_df = pd.merge(frames_df, player_stats_data_df, on='frame_num', how='left')
    player_stats_data_df = player_stats_data_df.ffill()

    # Add average speed calculations
    player_stats_data_df['player_1_average_shot_speed'] = player_stats_data_df['player_1_total_shot_speed'] / \
                                                          player_stats_data_df['player_1_number_of_shots']
    player_stats_data_df['player_2_average_shot_speed'] = player_stats_data_df['player_2_total_shot_speed'] / \
                                                          player_stats_data_df['player_2_number_of_shots']
    player_stats_data_df['player_1_average_player_speed'] = player_stats_data_df['player_1_total_player_speed'] / \
                                                            player_stats_data_df['player_2_number_of_shots']
    player_stats_data_df['player_2_average_player_speed'] = player_stats_data_df['player_2_total_player_speed'] / \
                                                            player_stats_data_df['player_1_number_of_shots']

    # Draw output
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections)
    output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)
    output_video_frames = mini_court.draw_mini_court(output_video_frames)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames, player_mini_court_detections)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames, ball_mini_court_detections,
                                                               color=(0, 255, 255))
    output_video_frames = draw_player_stats(output_video_frames, player_stats_data_df)

    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    save_video(output_video_frames, "output_videos/output_video3.avi")

if __name__ == "__main__":
    main()

import numpy as np
import cv2


def draw_player_stats(output_video_frames, player_stats):
    for index, row in player_stats.iterrows():
        # Extracting stats
        stats = {
            "player_1_shot_speed": row.get("player_1_last_shot_speed", 0),
            "player_2_shot_speed": row.get("player_2_last_shot_speed", 0),
            "player_1_speed": row.get("player_1_last_player_speed", 0),
            "player_2_speed": row.get("player_2_last_player_speed", 0),
            "player_1_distance": row.get("player_1_total_distance", 0),
            "player_2_distance": row.get("player_2_total_distance", 0),
            "player_1_stamina": row.get("player_1_stamina", 100),
            "player_2_stamina": row.get("player_2_stamina", 100),
            "player_1_avg_shot_speed": row.get("player_1_average_shot_speed", 0),
            "player_2_avg_shot_speed": row.get("player_2_average_shot_speed", 0),
            "player_1_avg_speed": row.get("player_1_average_player_speed", 0),
            "player_2_avg_speed": row.get("player_2_average_player_speed", 0),
            "player_1_favorite_side": row.get("player_1_favorite_side", "N/A"),
            "player_2_favorite_side": row.get("player_2_favorite_side", "N/A"),
        }

        frame = output_video_frames[index]
        overlay = frame.copy()

        # Dimensions for stats board (adjusted for position)
        board_x = frame.shape[1] - 420 + 20  # Slightly further right
        board_y = frame.shape[0] - 520 + 20  # Slightly further down
        board_width = 400
        board_height = 500

        # Apply transparency
        stats_board_color = (30, 30, 30)  # Dark background
        stats_board_alpha = 0.75

        cv2.rectangle(overlay, (board_x, board_y), (board_x + board_width, board_y + board_height), stats_board_color, -1)
        cv2.addWeighted(overlay, stats_board_alpha, frame, 1 - stats_board_alpha, 0, frame)

        # Title
        title_text = "Match Stats"
        cv2.putText(
            frame,
            title_text,
            (board_x + 100, board_y + 30),
            cv2.FONT_HERSHEY_DUPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        # Player labels
        cv2.putText(
            frame,
            "Player 1",
            (board_x + 160, board_y + 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 150, 255),  # Blue for Player 1
            2,
        )
        cv2.putText(
            frame,
            "Player 2",
            (board_x + 280, board_y + 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 150, 0),  # Orange for Player 2
            2,
        )

        # Stat categories and values
        stats_text = [
            ("Shot Speed", stats["player_1_shot_speed"], stats["player_2_shot_speed"], "km/h"),
            ("Player Speed", stats["player_1_speed"], stats["player_2_speed"], "km/h"),
            ("Avg Shot Speed", stats["player_1_avg_shot_speed"], stats["player_2_avg_shot_speed"], "km/h"),
            ("Avg Player Speed", stats["player_1_avg_speed"], stats["player_2_avg_speed"], "km/h"),
            ("Distance Covered", stats["player_1_distance"], stats["player_2_distance"], "m"),
            ("Favorite Side", stats["player_1_favorite_side"], stats["player_2_favorite_side"], ""),
        ]

        # Display stats in a grid format
        for i, (label, val1, val2, unit) in enumerate(stats_text):
            y_offset = board_y + 100 + i * 40
            label_color = (200, 200, 200)
            player_1_color = (0, 150, 255)  # Blue
            player_2_color = (255, 150, 0)  # Orange

            # Format numerical values to 2 decimal places
            if isinstance(val1, (float, int)):
                val1 = f"{val1:.2f}"
            if isinstance(val2, (float, int)):
                val2 = f"{val2:.2f}"

            # Stat name
            cv2.putText(
                frame,
                label,
                (board_x + 10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                label_color,
                1,
            )

            # Player 1 stat
            cv2.putText(
                frame,
                f"{val1} {unit}" if unit else val1,
                (board_x + 160, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                player_1_color,
                2,
            )

            # Player 2 stat
            cv2.putText(
                frame,
                f"{val2} {unit}" if unit else val2,
                (board_x + 280, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                player_2_color,
                2,
            )

        # Stamina bars
        bar_width = 200
        bar_height = 20
        bar_y1 = board_y + 380
        bar_y2 = board_y + 420

        def draw_stamina_bar(img, x, y, stamina, color, label):
            # Background bar
            cv2.rectangle(img, (x, y), (x + bar_width, y + bar_height), (50, 50, 50), -1)
            # Filled stamina
            filled_width = int(bar_width * stamina / 100)
            cv2.rectangle(img, (x, y), (x + filled_width, y + bar_height), color, -1)
            # Stamina label
            cv2.putText(
                img,
                label,
                (x + bar_width + 10, y + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

        # Player 1 stamina
        draw_stamina_bar(frame, board_x + 10, bar_y1, stats["player_1_stamina"], (0, 150, 255), "Stamina")

        # Player 2 stamina
        draw_stamina_bar(frame, board_x + 10, bar_y2, stats["player_2_stamina"], (255, 150, 0), "Stamina")

    return output_video_frames

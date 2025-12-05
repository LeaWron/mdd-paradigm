"""Run a sampling session using previously saved calibration models."""

import datetime
import sys
from pathlib import Path

import pygame
from pygame.locals import KEYDOWN, K_ESCAPE, K_RETURN


from gazefollower import GazeFollower
from gazefollower.calibration import SVRCalibration
from gazefollower.logger import Log
from gazefollower.misc import DefaultConfig


def main() -> None:
    pygame.init()
    screen_info = pygame.display.Info()
    win = pygame.display.set_mode(
        (screen_info.current_w, screen_info.current_h),
        pygame.FULLSCREEN | pygame.SCALED,
    )
    pygame.display.set_caption("GazeFollower Calibration + Sampling")

    model_dir = Path(__file__).resolve().parent
    model_dir = model_dir.joinpath("calib_models", "0_hrz")
    if not model_dir.exists():
        raise RuntimeError(
            f"Calibration model directory does not exist: {model_dir}. Please run the calibration script first."
        )

    calibration = SVRCalibration(model_save_path=str(model_dir))
    config = DefaultConfig()

    # gf_calibration = GazeFollower(calibration=calibration, config=config)
    # gf_calibration.preview(win=win)
    # gf_calibration.calibrate(win=win)

    # if not gf_calibration.calibration.has_calibrated:
    #     gf_calibration.release()
    #     pygame.quit()
    #     raise RuntimeError("Calibration failed. Check camera/logging output for details.")

    # if not gf_calibration.calibration.save_model():
    #     gf_calibration.release()
    #     pygame.quit()
    #     raise RuntimeError("Calibration trained but failed to save SVR models.")

    # gf_calibration.release()

    # Reload to prove persisted models can drive sampling.
    # reloaded_calibration = SVRCalibration(model_save_path=str(model_dir))
    # if not reloaded_calibration.has_calibrated:
    #     pygame.quit()
    #     raise RuntimeError("Failed to reload calibration from saved XML files.")

    gf = GazeFollower(calibration=calibration, config=config)
    gf.start_sampling()
    pygame.time.wait(200)

    background = pygame.Surface(win.get_size())
    background.fill((30, 30, 30))
    font = pygame.font.SysFont("Arial", 28)

    running = True
    clock = pygame.time.Clock()
    while running:
        for event in pygame.event.get():
            if event.type == KEYDOWN and event.key in {K_ESCAPE, K_RETURN}:
                running = False

        win.blit(background, (0, 0))

        gaze_info = gf.get_gaze_info()
        if (
            gaze_info
            and gaze_info.status
            and gaze_info.filtered_gaze_coordinates is not None
        ):
            gx = int(gaze_info.filtered_gaze_coordinates[0])
            gy = int(gaze_info.filtered_gaze_coordinates[1])
            pygame.draw.circle(win, (0, 255, 0), (gx, gy), 30, 4)
            pygame.draw.circle(win, (0, 128, 255), (gx, gy), 10)
            text = font.render(f"Gaze: ({gx}, {gy})", True, (255, 255, 255))
        else:
            text = font.render("Tracking...", True, (200, 200, 200))
        win.blit(text, (40, 40))

        pygame.display.flip()
        clock.tick(60)

    pygame.time.wait(200)
    gf.stop_sampling()

    data_dir = Path(__file__).resolve().parent / "data"
    data_dir.mkdir(exist_ok=True)
    output_csv = data_dir / "sampling_with_saved_model.csv"
    gf.save_data(str(output_csv))
    gf.release()

    pygame.quit()
    print(f"Sampling finished. Data saved to {output_csv}")


if __name__ == "__main__":
    main()

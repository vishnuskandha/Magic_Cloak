import cv2
import numpy as np
import time
from collections import deque


class BackgroundAnalyzer:
    """
    Maintains an adaptive background model using a running average with an optional
    freeze mask to avoid learning dynamic foreground (e.g., cloak region).
    """

    def __init__(self, alpha=0.02, stability_frames=60, enable_mask_freeze=True):
        self.alpha = float(alpha)
        self.stability_frames = int(stability_frames)
        self.enable_mask_freeze = bool(enable_mask_freeze)
        self.background_float = None
        self.frames_seen = 0
        # Static background support
        self.background_static = None
        self.use_static = False
        # Contamination control
        self.diff_threshold = 25  # per-channel approx after grayscale conversion
        self.max_diff_ratio_for_update = 0.15  # skip updating when too different

    def reset(self):
        self.background_float = None
        self.frames_seen = 0
        # keep static until explicitly cleared

    def update(self, frame_bgr, freeze_mask=None):
        frame_f32 = frame_bgr.astype(np.float32)
        if self.background_float is None:
            self.background_float = frame_f32.copy()
            self.frames_seen = 1
            return

        # Only update where allowed; prevent learning where cloak is detected
        mask_allowed = None
        if self.enable_mask_freeze and freeze_mask is not None:
            mask_allowed = cv2.bitwise_not(freeze_mask)
        # Prevent contamination by skipping large changes vs current background
        bg_uint8 = cv2.convertScaleAbs(self.background_float)
        gray_bg = cv2.cvtColor(bg_uint8, cv2.COLOR_BGR2GRAY)
        gray_fr = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray_bg, gray_fr)
        diff_mask = (diff > self.diff_threshold).astype(np.uint8) * 255
        if mask_allowed is None:
            update_mask = cv2.bitwise_not(diff_mask)
        else:
            update_mask = cv2.bitwise_and(mask_allowed, cv2.bitwise_not(diff_mask))
        # If too much is different, reduce learning rate locally
        ratio_diff = float(cv2.countNonZero(diff_mask)) / float(diff_mask.size)
        local_alpha = self.alpha * (0.25 if ratio_diff > self.max_diff_ratio_for_update else 1.0)
        cv2.accumulateWeighted(frame_f32, self.background_float, local_alpha, mask=update_mask)
        self.frames_seen += 1

    def is_stable(self):
        return self.frames_seen >= self.stability_frames

    def get_background(self):
        if self.use_static and self.background_static is not None:
            return self.background_static
        if self.background_float is None:
            return None
        return cv2.convertScaleAbs(self.background_float)

    def capture_static_from_frames(self, frames):
        if not frames:
            return False
        stack = np.stack(frames, axis=0)
        median = np.median(stack, axis=0).astype(np.uint8)
        self.background_static = median
        self.use_static = True
        return True

    def clear_static(self):
        self.background_static = None
        self.use_static = False


class CloakDetector:
    """
    Detects a target color in HSV space with temporal smoothing.
    Parameters are designed to be controlled via trackbars.
    """

    def __init__(self):
        self.smooth_factor = 0.6  # EMA blend for mask smoothing
        self.mask_ema = None

        # Defaults targeting a red cloak (optimized to avoid skin detection)
        self.h_center = 0
        self.h_width = 15
        self.s_min = 150  # Higher saturation to avoid skin tones
        self.s_max = 255
        self.v_min = 100  # Higher value to target bright red objects only
        self.v_max = 255

    def update_params(self, h_center, h_width, s_min, s_max, v_min, v_max, smooth_factor):
        self.h_center = int(max(0, min(179, h_center)))
        self.h_width = int(max(0, min(90, h_width)))
        self.s_min = int(max(0, min(255, s_min)))
        self.s_max = int(max(0, min(255, s_max)))
        self.v_min = int(max(0, min(255, v_min)))
        self.v_max = int(max(0, min(255, v_max)))
        self.smooth_factor = float(max(0.0, min(1.0, smooth_factor)))

    @staticmethod
    def preset_params(color_index):
        """
        Return (name, h_center, h_width, s_min, s_max, v_min, v_max) for common colors.
        Indices and defaults chosen for typical bright cloth colors.
        Red preset optimized to avoid skin tone detection.
        """
        presets = [
            ("Red", 0, 15, 150, 255, 100, 255),  # Optimized: Higher S_min (150) and V_min (100) to avoid skin
            ("Green", 60, 15, 100, 255, 50, 255),
            ("Blue", 120, 15, 100, 255, 50, 255),
            ("Yellow", 30, 12, 120, 255, 70, 255),
            ("Orange", 15, 12, 120, 255, 70, 255),
            ("Purple", 150, 15, 100, 255, 50, 255),
            ("Cyan", 90, 12, 100, 255, 50, 255),
            ("Magenta", 165, 12, 120, 255, 50, 255),
        ]
        idx = int(max(0, min(len(presets) - 1, color_index)))
        return presets[idx]

    def _hue_ranges(self):
        h_low = self.h_center - self.h_width
        h_high = self.h_center + self.h_width
        if h_low < 0:
            return [(h_low + 180, 179), (0, h_high)]
        if h_high > 179:
            return [(h_low, 179), (0, (h_high - 180))]
        return [(h_low, h_high)]

    def compute_mask(self, hsv_frame):
        # 1) Adaptive color range expansion based on current S/V
        s_mean = int(np.mean(hsv_frame[:, :, 1]))
        v_mean = int(np.mean(hsv_frame[:, :, 2]))
        s_min_adj = max(0, min(255, int(0.8 * self.s_min if s_mean > 150 else self.s_min)))
        v_min_adj = max(0, min(255, int(0.8 * self.v_min if v_mean > 150 else self.v_min)))

        masks = []
        for (hl, hh) in self._hue_ranges():
            lower = np.array([hl, s_min_adj, v_min_adj], dtype=np.uint8)
            upper = np.array([hh, self.s_max, self.v_max], dtype=np.uint8)
            masks.append(cv2.inRange(hsv_frame, lower, upper))
        mask = masks[0]
        if len(masks) == 2:
            mask = cv2.bitwise_or(masks[0], masks[1])

        # 2) Shadow suppression (lower V relative to surroundings often indicates shadows)
        v = hsv_frame[:, :, 2]
        blurred_v = cv2.GaussianBlur(v, (7, 7), 0)
        shadow_mask = (v < (blurred_v * 0.6)).astype(np.uint8) * 255
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(shadow_mask))

        # 3) Morphological cleanup with edge-aware refinement
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)

        # Edge refinement: keep mask where edges exist to avoid bleeding
        gray = cv2.cvtColor(cv2.cvtColor(hsv_frame, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 120)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        refined_mask = cv2.bitwise_and(mask, cv2.bitwise_or(mask, edges))

        # 4) Temporal smoothing via EMA + median over last 5 frames
        mask_f = refined_mask.astype(np.float32) / 255.0
        if self.mask_ema is None:
            self.mask_ema = mask_f
        else:
            sf = self.smooth_factor
            self.mask_ema = (sf * mask_f) + ((1.0 - sf) * self.mask_ema)
        mask_smoothed = (self.mask_ema > 0.5).astype(np.uint8) * 255
        return mask_smoothed


def create_controls():
    # Recreate controls window and trackbars from scratch
    try:
        if cv2.getWindowProperty("Controls", cv2.WND_PROP_VISIBLE) >= 0:
            cv2.destroyWindow("Controls")
    except cv2.error:
        pass
    cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Controls", 480, 300)
    # Preset color selector (0..7) and manual toggle
    cv2.createTrackbar("Preset", "Controls", 0, 7, lambda v: None)
    cv2.createTrackbar("ManualHSV", "Controls", 0, 1, lambda v: None)
    # Hue controls
    cv2.createTrackbar("H_center", "Controls", 0, 179, lambda v: None)
    cv2.createTrackbar("H_width", "Controls", 20, 90, lambda v: None)
    # Saturation and value (optimized defaults to avoid skin detection)
    cv2.createTrackbar("S_min", "Controls", 150, 255, lambda v: None)  # Higher default S_min
    cv2.createTrackbar("S_max", "Controls", 255, 255, lambda v: None)
    cv2.createTrackbar("V_min", "Controls", 100, 255, lambda v: None)  # Higher default V_min
    cv2.createTrackbar("V_max", "Controls", 255, 255, lambda v: None)
    # Background learning rate and smoothing
    cv2.createTrackbar("Alpha(%)", "Controls", 2, 100, lambda v: None)
    cv2.createTrackbar("Smooth(%)", "Controls", 60, 100, lambda v: None)
    # Preservation toggles
    cv2.createTrackbar("PreserveFace", "Controls", 1, 1, lambda v: None)
    cv2.createTrackbar("PreserveSkin", "Controls", 1, 1, lambda v: None)
    # Give HighGUI a moment to realize the window/trackbars
    for _ in range(3):
        cv2.waitKey(1)


def ensure_controls_window_exists():
    try:
        visible = cv2.getWindowProperty("Controls", cv2.WND_PROP_VISIBLE)
    except cv2.error:
        visible = -1
    if visible < 0:
        create_controls()
        for _ in range(3):
            cv2.waitKey(1)


def read_controls():
    preset_idx = cv2.getTrackbarPos("Preset", "Controls")
    manual_hsv = cv2.getTrackbarPos("ManualHSV", "Controls")
    h_center = cv2.getTrackbarPos("H_center", "Controls")
    h_width = cv2.getTrackbarPos("H_width", "Controls")
    s_min = cv2.getTrackbarPos("S_min", "Controls")
    s_max = cv2.getTrackbarPos("S_max", "Controls")
    v_min = cv2.getTrackbarPos("V_min", "Controls")
    v_max = cv2.getTrackbarPos("V_max", "Controls")
    alpha_pct = cv2.getTrackbarPos("Alpha(%)", "Controls")
    smooth_pct = cv2.getTrackbarPos("Smooth(%)", "Controls")
    preserve_face = cv2.getTrackbarPos("PreserveFace", "Controls")
    preserve_skin = cv2.getTrackbarPos("PreserveSkin", "Controls")
    alpha = max(0.001, min(1.0, alpha_pct / 100.0))
    smooth = max(0.0, min(1.0, smooth_pct / 100.0))
    return preset_idx, manual_hsv, h_center, h_width, s_min, s_max, v_min, v_max, alpha, smooth, preserve_face, preserve_skin


def compute_skin_mask_ycrcb(frame_bgr):
    ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    # Typical skin range in YCrCb
    skin = cv2.inRange(ycrcb, (0, 135, 85), (255, 180, 135))
    # Clean up
    kernel = np.ones((3, 3), np.uint8)
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, kernel, iterations=1)
    skin = cv2.morphologyEx(skin, cv2.MORPH_DILATE, kernel, iterations=1)
    return skin


def color_picker_callback(event, x, y, flags, param):
    """Mouse callback for color picking"""
    if event == cv2.EVENT_LBUTTONDOWN:
        frame, detector = param
        if frame is not None:
            # Get color at click point
            bgr_color = frame[y, x]
            hsv_color = cv2.cvtColor(np.uint8([[bgr_color]]), cv2.COLOR_BGR2HSV)[0][0]
            h, s, v = hsv_color
            
            # Set trackbars to detected color with some tolerance
            cv2.setTrackbarPos("H_center", "Controls", int(h))
            cv2.setTrackbarPos("H_width", "Controls", 25)  # Wider tolerance
            cv2.setTrackbarPos("S_min", "Controls", max(0, int(s - 30)))
            cv2.setTrackbarPos("S_max", "Controls", min(255, int(s + 30)))
            cv2.setTrackbarPos("V_min", "Controls", max(0, int(v - 40)))
            cv2.setTrackbarPos("V_max", "Controls", min(255, int(v + 40)))
            cv2.setTrackbarPos("ManualHSV", "Controls", 1)  # Switch to manual mode
            print(f"Color picked: BGR={bgr_color}, HSV={hsv_color}")


def analyze_color_distribution(frame, mask):
    """Analyze the color distribution in the masked region"""
    if np.sum(mask) == 0:
        return None
    
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
    hsv = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2HSV)
    
    # Get non-zero pixels
    hsv_pixels = hsv[mask > 0]
    if len(hsv_pixels) == 0:
        return None
    
    h, s, v = hsv_pixels[:, 0], hsv_pixels[:, 1], hsv_pixels[:, 2]
    
    # Calculate statistics
    h_mean, h_std = np.mean(h), np.std(h)
    s_mean, s_std = np.mean(s), np.std(s)
    v_mean, v_std = np.mean(v), np.std(v)
    
    return {
        'h_mean': h_mean, 'h_std': h_std,
        's_mean': s_mean, 's_std': s_std,
        'v_mean': v_mean, 'v_std': v_std,
        'pixel_count': len(hsv_pixels)
    }


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera.")
        return

    # Optional: set a reasonable resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    create_controls()
    detector = CloakDetector()
    bg = BackgroundAnalyzer(alpha=0.02, stability_frames=60, enable_mask_freeze=True)

    print("Controls: q-quit | r-reset bg | b-freeze bg | g-grab bg | s-save | 0-7 preset | m manual HSV | f face | k skin | c color picker")
    print("Click on your cloak in the main window to auto-detect color!")

    # Warmup
    time.sleep(1.0)

    freeze_enabled = True
    grab_static_frames = 0
    static_capture_mode = False
    static_buffer = []
    static_buffer_target = 45  # frames for median capture
    color_picker_mode = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = np.flip(frame, axis=1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        ensure_controls_window_exists()
        try:
            preset_idx, manual_hsv, h_center, h_width, s_min, s_max, v_min, v_max, alpha, smooth, preserve_face, preserve_skin = read_controls()
        except cv2.error:
            # If for any reason the Controls window vanished mid-run, recreate and continue
            create_controls()
            for _ in range(3):
                cv2.waitKey(1)
            preset_idx, manual_hsv, h_center, h_width, s_min, s_max, v_min, v_max, alpha, smooth, preserve_face, preserve_skin = read_controls()
        if manual_hsv:
            detector.update_params(h_center, h_width, s_min, s_max, v_min, v_max, smooth)
            preset_name = "Manual"
        else:
            preset_name, phc, phw, psmin, psmax, pvmin, pvmax = CloakDetector.preset_params(preset_idx)
            detector.update_params(phc, phw, psmin, psmax, pvmin, pvmax, smooth)
        bg.alpha = alpha

        mask = detector.compute_mask(hsv)
        # Subtract face/skin to avoid hiding the face
        if preserve_skin:
            skin_mask = compute_skin_mask_ycrcb(frame)
            mask = cv2.bitwise_and(mask, cv2.bitwise_not(skin_mask))

        # Optionally freeze learning where cloak is detected
        freeze_mask = mask if freeze_enabled else None

        # Static background capture workflow
        if static_capture_mode:
            static_buffer.append(frame.copy())
            if len(static_buffer) >= static_buffer_target:
                bg.capture_static_from_frames(static_buffer)
                static_buffer = []
                static_capture_mode = False
                print("Static background captured.")
        elif grab_static_frames > 0:
            # Adaptive quick grab across a short window
            freeze_mask = None  # learn everywhere for static grab
            grab_static_frames -= 1

        bg.update(frame, freeze_mask=freeze_mask)
        background = bg.get_background()
        if background is None:
            background = frame

        inv_mask = cv2.bitwise_not(mask)
        cloak_region = cv2.bitwise_and(background, background, mask=mask)
        foreground_region = cv2.bitwise_and(frame, frame, mask=inv_mask)
        output = cv2.add(cloak_region, foreground_region)

        # Compose a small HUD with status
        hud = output.copy()
        status = f"Preset: {preset_name} | BG stable: {bg.is_stable()} | Freeze: {freeze_enabled} | Face:{bool(preserve_face)} Skin:{bool(preserve_skin)} | a={alpha:.2f} sm={smooth:.2f}"
        if color_picker_mode:
            status += " | CLICK TO PICK COLOR"
        cv2.putText(hud, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 220, 20), 2, cv2.LINE_AA)

        # Set up color picker callback
        if color_picker_mode:
            cv2.setMouseCallback("Magic Cloak", color_picker_callback, (frame, detector))

        cv2.imshow("Magic Cloak", hud)
        cv2.imshow("Mask", mask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            bg.reset()
        elif key == ord('b'):
            freeze_enabled = not freeze_enabled
        elif key == ord('g'):
            # Grab a clean static background quickly
            bg.reset()
            grab_static_frames = 45
        elif key == ord('G'):
            # Enter static capture mode (median over many frames)
            bg.clear_static()
            static_buffer = []
            static_capture_mode = True
            print("Static capture mode ON - ensure scene is empty.")
        elif key == ord('u'):
            # Use static background (if captured)
            bg.use_static = True
            print("Using static background if available.")
        elif key == ord('y'):
            # Use adaptive background
            bg.use_static = False
            print("Using adaptive background.")
        elif key == ord('s'):
            timestamp = int(time.time())
            cv2.imwrite(f"cloak_{timestamp}.png", hud)
        elif key in [ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7')]:
            idx = int(chr(key))
            cv2.setTrackbarPos("Preset", "Controls", idx)
            cv2.setTrackbarPos("ManualHSV", "Controls", 0)
        elif key == ord('m'):
            current = cv2.getTrackbarPos("ManualHSV", "Controls")
            cv2.setTrackbarPos("ManualHSV", "Controls", 0 if current else 1)
        elif key == ord('f'):
            cur = cv2.getTrackbarPos("PreserveFace", "Controls")
            cv2.setTrackbarPos("PreserveFace", "Controls", 0 if cur else 1)
        elif key == ord('k'):
            cur = cv2.getTrackbarPos("PreserveSkin", "Controls")
            cv2.setTrackbarPos("PreserveSkin", "Controls", 0 if cur else 1)
        elif key == ord('c'):
            color_picker_mode = not color_picker_mode
            if color_picker_mode:
                print("Color picker ON - Click on your cloak to detect color!")
            else:
                print("Color picker OFF")
                cv2.setMouseCallback("Magic Cloak", lambda *args: None)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

# generator/utils/sampling_timer.py

import time


class SamplingTimer:
    """
    High‑precision drift‑corrected timer for telemetry sampling loops.

    Purpose:
        Ensures that each iteration of the generator loop runs at the
        configured sampling frequency (Hz) without accumulating drift.

    Example:
        timer = SamplingTimer(frequency_hz=10)
        while generating:
            timer.sleep_until_next_tick()
            generate_next_sample()

    Parameters:
        frequency_hz: float
            Sampling frequency in Hertz (cycles per second)
    """

    def __init__(self, frequency_hz: float):
        self.frequency_hz = max(0.0001, float(frequency_hz))  # avoid division by zero
        self.interval = 1.0 / self.frequency_hz
        self.next_tick = time.perf_counter()

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------
    def sleep_until_next_tick(self):
        """
        Sleeps until the next scheduled tick time.
        Automatically corrects for drift.

        If the generator loop is slow and falls behind,
        the timer will skip missed intervals to catch up.
        """
        now = time.perf_counter()

        # If we are behind schedule, skip ahead
        if now > self.next_tick:
            self.next_tick = now + self.interval
            return

        # Otherwise sleep until the next tick
        sleep_time = self.next_tick - now
        if sleep_time > 0:
            time.sleep(sleep_time)

        # Schedule next tick
        self.next_tick += self.interval

    def reset(self):
        """
        Resets the timer to start counting from now.
        """
        self.next_tick = time.perf_counter()
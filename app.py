import threading
import tkinter as tk
from tkinter import messagebox

from app.model import Model
from app.view import View


class Controller:
    """Controller class."""

    def __init__(self):
        """Initialize model, view and Tk root window."""
        self.m = Model(self)
        self.root = tk.Tk()
        self.view = View(self.root, self)
        self._is_training = False

    def run(self):
        """Start Tkinter event loop."""
        self.root.mainloop()

    def do_train(self):
        """Run training in a background thread."""
        if self._is_training:
            messagebox.showwarning(title="Training", message="Training is already running.")
            return

        self._set_training_state(True)
        worker = threading.Thread(target=self._train_worker, daemon=True)
        worker.start()

    def _train_worker(self):
        try:
            self.m.train(progress_callback=self._on_train_progress)
        except Exception as exc:
            self.root.after(0, lambda err=str(exc): self._on_train_error(err))
            return

        self.root.after(0, self._on_train_success)

    def _on_train_error(self, error_message):
        self._set_training_state(False)
        messagebox.showerror(title="Training Error", message=error_message)

    def _on_train_success(self):
        self._set_training_state(False)
        self.view.set_status("Training completed.")

    def _set_training_state(self, training):
        self._is_training = training
        self.view.set_train_enabled(not training)
        if training:
            self.view.set_status("Training started...")

    def _on_train_progress(self, epoch, total_epochs, loss, eta_seconds):
        """Print training progress with ETA and update UI status."""
        eta_min, eta_sec = divmod(max(0, int(eta_seconds)), 60)
        loss_txt = "n/a" if loss is None else f"{loss:.6f}"
        status = (
            f"Training {epoch}/{total_epochs} | "
            f"loss={loss_txt} | ETA {eta_min:02d}:{eta_sec:02d}"
        )
        print(f"[TRAIN] {status}")
        self.root.after(0, lambda s=status: self.view.set_status(s))

    def do_plot(self):
        """Display plot after training."""
        if self._is_training:
            messagebox.showwarning(title="Training", message="Wait for training to finish.")
            return

        if not self.m.ready_to_do_plot:
            messagebox.showwarning(title="Wrong Button!", message="You need to train first")
            return

        try:
            self.m.plot()
        except Exception as exc:
            messagebox.showerror(title="Plot Error", message=str(exc))

    def do_gain(self):
        """Calculate gain and display it."""
        if self._is_training:
            messagebox.showwarning(title="Training", message="Wait for training to finish.")
            return

        try:
            gain = self.m.gain()
        except Exception as exc:
            messagebox.showerror(title="Gain Error", message=str(exc))
            return

        messagebox.showinfo(
            title="Gain",
            message=(
                f"Current value is {int(gain)}% of starting value\n"
                f"100% - Gain is equal: {100 - int(gain)}%\n"
            ),
        )

    def set_model_currency(self, i):
        self.m.set_currency(i)

    def set_model_data_source(self, i):
        supported = self.m.set_data_source(i)
        if not supported:
            messagebox.showwarning(
                title="Unsupported Source",
                message="Selected source is unsupported for crypto. Switched to Yahoo.",
            )
            self.view.set_data_source(1)

    def set_model_model_id(self, i):
        self.m.set_model_id(i)

    def set_model_model_ID(self, i):
        """Backward-compatible alias."""
        self.set_model_model_id(i)

    def set_model_file_path(self, path):
        self.m.set_file_path(path)

    def set_model_switch_state(self, i):
        self.m.set_switch_state(i)

    def set_model_prediction_days(self, i):
        self.m.set_prediction_days(i)

    def set_model_predition_days(self, i):
        """Backward-compatible alias for typo in old API."""
        self.set_model_prediction_days(i)

    def set_model_future_days(self, i):
        self.m.set_future_days(i)

    def set_model_test_start_date(self, i):
        self.m.set_test_start_date(i)


if __name__ == "__main__":
    Controller().run()

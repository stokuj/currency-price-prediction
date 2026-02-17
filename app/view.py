import os
import tkinter as tk
from tkinter import filedialog as fd
from tkinter import messagebox
from tkinter import ttk


class View:
    """View class used to build Tkinter GUI."""

    def __init__(self, root, controller):
        self.root = root
        self.controller = controller

        root.title("Cryptocurrency Price Prediction")

        window_height = 520
        window_width = 480

        def center_screen():
            screen_width = root.winfo_screenwidth()
            screen_height = root.winfo_screenheight()
            x_coordinate = int((screen_width / 2) - (window_width / 2))
            y_coordinate = int((screen_height / 2) - (window_height / 2))
            root.geometry(f"{window_width}x{window_height}+{x_coordinate}+{y_coordinate}")

        center_screen()

        style = ttk.Style(root)
        root.tk.call("source", os.path.join(os.path.dirname(__file__), "azure.tcl"))
        style.theme_use("azure")

        currency_type = tk.IntVar(value=1)
        self.data_source = tk.IntVar(value=1)
        future_day = tk.StringVar(value="1")
        test_start_date = tk.StringVar(value="20")
        prediction_days = tk.IntVar(value=30)
        switch_state = tk.IntVar(value=1)  # 1=online, 0=offline
        model_id = tk.IntVar(value=1)
        self.status_var = tk.StringVar(value="Idle")

        frame1 = ttk.LabelFrame(root, text="Select Currency", width=210, height=200)
        frame1.place(x=20, y=12)

        frame2 = ttk.LabelFrame(root, text="Online Source", width=210, height=160)
        frame2.place(x=20, y=252)

        frame3 = ttk.LabelFrame(root, text="Select Model", width=210, height=160)
        frame3.place(x=250, y=252)

        ttk.Radiobutton(frame1, text="BTC", variable=currency_type, value=1).place(x=20, y=20)
        ttk.Radiobutton(frame1, text="ETH", variable=currency_type, value=2).place(x=20, y=60)
        ttk.Radiobutton(frame1, text="Doge", variable=currency_type, value=3).place(x=20, y=100)
        ttk.Radiobutton(frame1, text="LTC", variable=currency_type, value=4).place(x=20, y=140)

        def on_data_source_change():
            self.controller.set_model_data_source(self.data_source.get())

        ttk.Radiobutton(
            frame2, text="Yahoo", variable=self.data_source, value=1, command=on_data_source_change
        ).place(x=20, y=20)
        ttk.Radiobutton(
            frame2, text="Stooq", variable=self.data_source, value=2, command=on_data_source_change
        ).place(x=20, y=60)
        ttk.Radiobutton(
            frame2, text="Naver", variable=self.data_source, value=3, command=on_data_source_change
        ).place(x=20, y=100)

        ttk.Radiobutton(frame3, text="LSTM", variable=model_id, value=1).place(x=20, y=20)
        ttk.Radiobutton(frame3, text="GRU", variable=model_id, value=2).place(x=20, y=60)
        ttk.Radiobutton(frame3, text="LSTM+GRU", variable=model_id, value=3).place(x=20, y=100)

        def get_current_value():
            return " Based on last {: .0f}days".format(prediction_days.get())

        value_label1 = ttk.Label(root, text="Based on last 30 days")
        value_label1.place(x=360, y=60)

        def on_scale(_):
            prediction_days.set(int(scale_widget.get()))
            value_label1.configure(text=get_current_value())
            self.controller.set_model_prediction_days(prediction_days.get())

        scale_widget = ttk.Scale(root, from_=1, to=100, variable=prediction_days, command=on_scale)
        scale_widget.place(x=250, y=20)

        ttk.Progressbar(root, value=1, variable=prediction_days, mode="determinate").place(x=250, y=60)
        ttk.Label(root, text="Prediction:").place(x=360, y=20)

        ttk.Spinbox(root, from_=1, to=10, increment=1, textvariable=future_day).place(x=250, y=140)
        ttk.Spinbox(root, from_=20, to=600, increment=5, textvariable=test_start_date).place(x=250, y=180)

        ttk.Label(root, text="Future days").place(x=400, y=145)
        ttk.Label(root, text="Plot range").place(x=400, y=185)

        self.file_label = ttk.Label(root, text="")
        self.file_label.place(x=120, y=425)

        def callback_button1():
            filename = fd.askopenfilename()
            if not filename:
                return
            self.file_label.config(text=filename)
            self.controller.set_model_file_path(filename)

        self.file_button = ttk.Button(root, text="Select file", command=callback_button1)
        self.file_button.place(x=20, y=430)

        def callback_button2():
            self.controller.set_model_currency(currency_type.get())
            self.controller.set_model_model_id(model_id.get())
            self.controller.set_model_switch_state(switch_state.get())
            self.controller.set_model_prediction_days(prediction_days.get())
            self.controller.set_model_future_days(int(future_day.get()))
            self.controller.set_model_test_start_date(int(test_start_date.get()))
            self.controller.do_train()

        self.train_button = ttk.Button(root, text="Train", command=callback_button2)
        self.train_button.place(x=20, y=480)
        ttk.Button(root, text="Plot", command=self.controller.do_plot).place(x=120, y=480)

        def callback_button4():
            messagebox.showinfo(
                title="Help",
                message=(
                    "Welcome in Crypto Currency Prediction program\n\n"
                    "Plot range  - Number of dated points shown on chart\n"
                    "Future days - Forecast horizon in days (prediction offset)\n"
                    "Prediction days - Lookback window size used by the model\n"
                ),
            )

        ttk.Button(root, text="HELP", command=callback_button4).place(x=380, y=480)
        ttk.Button(root, text="GAIN", command=self.controller.do_gain).place(x=380, y=430)

        ttk.Label(root, textvariable=self.status_var).place(x=20, y=455)

        def apply_online_state():
            is_online = switch_state.get() == 1
            self.controller.set_model_switch_state(switch_state.get())
            self.file_button.config(state=tk.DISABLED if is_online else tk.NORMAL)

        switch = ttk.Checkbutton(
            root,
            text="Use online database",
            style="Switch",
            variable=switch_state,
            offvalue=0,
            onvalue=1,
            command=apply_online_state,
        )
        switch.place(x=250, y=100)

        # Initialize control/model state.
        apply_online_state()
        on_data_source_change()

        ttk.Sizegrip(root).place(x=780, y=510)
        ttk.Separator().place(x=20, y=235, width=210)
        ttk.Separator().place(x=250, y=235, width=210)

    def set_train_enabled(self, enabled):
        self.train_button.config(state=tk.NORMAL if enabled else tk.DISABLED)

    def set_status(self, text):
        self.status_var.set(text)

    def set_data_source(self, value):
        self.data_source.set(int(value))

from statistics import variance
import tkinter as tk
from tkinter import Variable, ttk
from tkinter import filedialog as fd
from tkinter import messagebox


class View():
    """ View class, used to make gui with tkinter
    """
    def __init__(self, root, controller):
        self.root = root
        self.controller = controller
        
        ### Title definition
        root.title('Cryptocurrency Price Prediction')
        
        start_bool = True;
        ### Size fo window and function to center window
        window_height = 520
        window_width = 480
        def center_screen():
            """ gets the coordinates of the center of the screen """
            global screen_height, screen_width, x_cordinate, y_cordinate

            screen_width = root.winfo_screenwidth()
            screen_height = root.winfo_screenheight()
            x_cordinate = int((screen_width/2) - (window_width/2))
            y_cordinate = int((screen_height/2) - (window_height/2))
            root.geometry("{}x{}+{}+{}".format(window_width, window_height, x_cordinate, y_cordinate))

        center_screen()

        ### Set style and theme of tkinter
        style = ttk.Style(root)
        root.tk.call('source', 'azure.tcl')
        style.theme_use('azure')

        ### Set up variabled used in view
        currency_type = tk.IntVar()
        currency_type.set(1)
        b = tk.IntVar()
        b.set(1)
        c = tk.IntVar()
        data_source = tk.IntVar()
        data_source.set(1)
        f = tk.IntVar()
        
        future_day = tk.StringVar()
        future_day.set('1')
        test_start_date = tk.StringVar()
        test_start_date.set('20')

        prediction_days = tk.IntVar()
        prediction_days.set(30)

        switch_state = tk.IntVar()
        model_id = tk.IntVar()
        model_id.set(1)

        
        ####################################################################################################
        ### Frame creation
        
        ### Left up
        frame1 = ttk.LabelFrame(root, text='Select Currency', width=210, height=200)
        frame1.place(x=20, y=12)

        ### Left down
        frame2 = ttk.LabelFrame(root, text='Online Source', width=210, height=160)
        frame2.place(x=20, y=252)

        ### Right down
        frame3 = ttk.LabelFrame(root, text='Select Model', width=210, height=160)
        frame3.place(x=250, y=252)
        
        ### Radiobuttons in frame1, Left Up
        check1 = ttk.Radiobutton(frame1, text='BTC', variable=currency_type, value=1)
        check1.place(x=20, y=20)
        check2 = ttk.Radiobutton(frame1, text='ETH', variable=currency_type, value=2)
        check2.place(x=20, y=60)
        check3 = ttk.Radiobutton(frame1, text='Doge', variable=currency_type, value=3)
        check3.place(x=20, y=100)
        check4 = ttk.Radiobutton(frame1, text='LTC', variable=currency_type, value=4)
        check4.place(x=20, y=140)

        ### Radiobuttons in frame2, Left Down
        radio1 = ttk.Radiobutton(frame2, text='Yahoo', variable=data_source, value=1)
        radio1.place(x=20, y=20)
        radio2 = ttk.Radiobutton(frame2, text='Stooq', variable=data_source, value=2)
        radio2.place(x=20, y=60)
        radio3 = ttk.Radiobutton(frame2, text='Naver', variable=data_source, value=3)
        radio3.place(x=20, y=100)
        
        ### Radiobuttons in frame2, Right Down
        radio4 = ttk.Radiobutton(frame3, text='LSTM', variable=model_id, value=1)
        radio4.place(x=20, y=20)
        radio5 = ttk.Radiobutton(frame3, text='GRU', variable=model_id, value=2)
        radio5.place(x=20, y=60)
        radio6 = ttk.Radiobutton(frame3, text='LSTM+GRU', variable=model_id, value=3)
        radio6.place(x=20, y=100)

        ####################################################################################################
        ### Elements without frame
        
        ### Elements in right up corner area - Scale with progress and two labels
        
        def scale(i):
            prediction_days.set(int(scale.get()))
            value_label1.configure(text=get_current_value())
            self.controller.set_model_predition_days(prediction_days.get())

        scale = ttk.Scale(root, from_=1, to=100, variable=prediction_days, command=scale)
        scale.place(x=250, y=20)

        progress = ttk.Progressbar(root, value=1, variable=prediction_days, mode='determinate')
        progress.place(x=250, y=60)

        value_label2 = ttk.Label( root, text = 'Prediction:')
        value_label2.place(x=360, y=20)
        
        value_label1 = ttk.Label( root, text = 'Based on last 30 days')
        value_label1.place(x=360, y=60)
        
        ####################################################################################################
        ### Elements without frame
        
        ### Spinbox for futureday and range of chart and two labers for them

        spin1 = ttk.Spinbox(root, from_= 1, to=10, increment=1, textvariable = future_day)
        spin1.place(x=250, y=140)

        spin2 = ttk.Spinbox(root, from_= 20, to=600, increment=5, textvariable = test_start_date)
        spin2.place(x=250, y=180)

        labael_spin1 = ttk.Label( root, text = 'Future days')
        labael_spin1.place(x=400, y=145)
        
        labael_spin2 = ttk.Label( root, text = 'Plot range')
        labael_spin2.place(x=400, y=185)
        
        ####################################################################################################
        ### Buttons from left to right below frame 2 and 3
        
        def callback_button1():
            filename = fd.askopenfilename()
            laber = ttk.Label(root, text = filename)
            laber.place(x=120, y=425)
            self.controller.set_model_file_path(filename)

        ### Callback and button to select the file when switch is off
        ###
        button1 = ttk.Button(root, text='Select file', command=callback_button1, state = tk.NORMAL)
        button1.place(x=20, y=430)

        def callback_button2():
            self.controller.set_model_currency(currency_type.get())
            self.controller.set_model_data_source(data_source.get())
            self.controller.set_model_model_ID(model_id.get())
            self.controller.set_model_switch_state(switch_state.get())
            self.controller.set_model_predition_days(prediction_days.get())
            self.controller.set_model_future_days(int(future_day.get()))
            self.controller.set_model_test_start_date(int(test_start_date.get()))
            self.controller.do_train()
            
        ### Callback and button to display Train
        ###
        button2 = ttk.Button(root, text='Train', command = callback_button2)
        button2.place(x=20, y=480)

        def callback_button3():
            self.controller.do_plot()
            
        ### Callback and button to display Plot
        ###
        self.button3 = ttk.Button(root, text='Plot', command = callback_button3)
        self.button3.place(x=120, y=480)
        
        ### Callback and button to display HELP
        ###
        def callback_button4():
            messagebox.showinfo(title='Help', 
                                message = 'Welcome in Crypto Currency Precidion program\n\n'
                                +   'Plot range  - Size of chart up to 600 days\n'
                                +   'Future days - Range of prediction to future\n'
                                +   'Prediction  - On how many days point should be predicted\n'
                                )
        
        self.button4 = ttk.Button(root, text='HELP', command = callback_button4)
        self.button4.place(x=380, y=480)
        
        ### Callback and button to display gain
        ###
        def callback_button5():
            self.controller.do_gain()
            
        self.button5 = ttk.Button(root, text='GAIN', command = callback_button5)
        self.button5.place(x=380, y=430)
        
        ####################################################################################################
        ### Switch and separator abowe frame 2

        def callback_switch():
            if (button1['state'] == tk.NORMAL):
                button1['state'] =  tk.DISABLED
            else:
                button1['state'] =  tk.NORMAL
                
                
        switch = ttk.Checkbutton(root, text='Use online database', style='Switch', variable=switch_state, offvalue=0, onvalue=1, command = callback_switch)
        switch.place(x=250, y=100)
        switch.invoke()

        ### Sperators

        size = ttk.Sizegrip(root)
        size.place(x=780, y=510)
        
        sep1 = ttk.Separator()
        sep1.place(x=20, y=235, width=210)

        sep2 = ttk.Separator()
        sep2.place(x=250, y=235, width=210)

        def get_current_value():
            """ Function gest current value from predicted day variable

            Returns:
                string: How many days of predigion is being set
            """
            return ' Based on last {: .0f}days'.format(prediction_days.get())
        
    
    

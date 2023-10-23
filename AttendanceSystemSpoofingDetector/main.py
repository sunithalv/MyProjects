import os.path
import datetime
import pickle
import subprocess

import tkinter as tk
import cv2
from PIL import Image, ImageTk
import face_recognition

import util
from test import test


class App:
    def __init__(self):
        self.main_window = tk.Tk()
        self.main_window.geometry("1200x520+350+100")
        #Call function login on click of login button
        self.login_button_main_window = util.get_button(self.main_window, 'login', 'green', self.login)
        self.login_button_main_window.place(x=750, y=200)

        self.logout_button_main_window = util.get_button(self.main_window, 'logout', 'red', self.logout)
        self.logout_button_main_window.place(x=750, y=300)

        # Call register_new_user function on click of the button
        self.register_new_user_button_main_window = util.get_button(self.main_window, 'register new user', 'gray',
                                                                    self.register_new_user, fg='black')
        self.register_new_user_button_main_window.place(x=750, y=400)

        #Webcam window
        self.webcam_label = util.get_img_label(self.main_window)
        self.webcam_label.place(x=10, y=0, width=700, height=500)

        #Call function add_webcam to display the webcam
        self.add_webcam(self.webcam_label)

        #Create new directory to store the registered users
        self.db_dir = './db'
        if not os.path.exists(self.db_dir):
            os.mkdir(self.db_dir)
        #Log file details
        self.log_path = './log.txt'

    def add_webcam(self, label):
        #Since we will be accessing the function multiple times check if the variable is already created
        if 'cap' not in self.__dict__:
            self.cap = cv2.VideoCapture(0)

        self._label = label
        self.process_webcam()

    #Puts the webcam into the label
    def process_webcam(self):
        ret, frame = self.cap.read()

        self.most_recent_capture_arr = frame
        img_ = cv2.cvtColor(self.most_recent_capture_arr, cv2.COLOR_BGR2RGB)
        #get most recent captured image
        self.most_recent_capture_pil = Image.fromarray(img_)
        imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
        self._label.imgtk = imgtk
        self._label.configure(image=imgtk)

        self._label.after(20, self.process_webcam)

    def login(self):
        # unknown_img_path='./.tmp.jpg'
        # cv2.imwrite(unknown_img_path,self.most_recent_capture_arr)
        # #Run the face_recognition command to check the known/unknown
        # output=str(subprocess.check_output(['face_recognition',self.db_dir,unknown_img_path]))
        # name=output.split(',')[1][:-5]
        # if name in ['unknown_person','no_persons_found']:
        #     util.msg_box('Unknown user','Oops...Unknown user,Please register new user or try again')
        # else:
        #     util.msg_box('Welcome','Welcome back,{}'.format(name))
        #     with open(self.log_path,'a') as f:
        #         #Log the user name and time
        #         f.write('{}.{}\n'.format(name,datetime.datetime.now()))
        #
        #
        #
        # os.remove(unknown_img_path)
        #Check if image is fake using the antispoof library installed
        label = test(
                image=self.most_recent_capture_arr,
                model_dir='C:/Users/sunit/Desktop/LATEST_PROJECTS/AttendanceSystem/Silent-Face-Anti-Spoofing-master/resources/anti_spoof_models',
                device_id=0
                )
        #1 indicates real image;0 is fake
        print('OUTPUT:',label)
        if label == 1:

            name = util.recognize(self.most_recent_capture_arr, self.db_dir)

            if name in ['unknown_person', 'no_persons_found']:
                util.msg_box('Ups...', 'Unknown user. Please register new user or try again.')
            else:
                util.msg_box('Welcome back !', 'Welcome, {}.'.format(name))
                with open(self.log_path, 'a') as f:
                    f.write('{},{},in\n'.format(name, datetime.datetime.now()))
                    f.close()

        else:
            util.msg_box('Hey, you are a spoofer!', 'You are fake !')

    def logout(self):

        label = test(
                image=self.most_recent_capture_arr,
                model_dir='C:/Users/sunit/Desktop/LATEST_PROJECTS/AttendanceSystem/Silent-Face-Anti-Spoofing-master/resources/anti_spoof_models',
                device_id=0
                )

        if label == 1:

            name = util.recognize(self.most_recent_capture_arr, self.db_dir)

            if name in ['unknown_person', 'no_persons_found']:
                util.msg_box('Ups...', 'Unknown user. Please register new user or try again.')
            else:
                util.msg_box('Hasta la vista !', 'Goodbye, {}.'.format(name))
                with open(self.log_path, 'a') as f:
                    f.write('{},{},out\n'.format(name, datetime.datetime.now()))
                    f.close()

        else:
            util.msg_box('Hey, you are a spoofer!', 'You are fake !')


    def register_new_user(self):
        self.register_new_user_window = tk.Toplevel(self.main_window)
        self.register_new_user_window.geometry("1200x520+370+120")
        #Call function accept_register_new_user on click
        self.accept_button_register_new_user_window = util.get_button(self.register_new_user_window, 'Accept', 'green', self.accept_register_new_user)
        self.accept_button_register_new_user_window.place(x=750, y=300)
        # Call function try_again_register_new_user on click
        self.try_again_button_register_new_user_window = util.get_button(self.register_new_user_window, 'Try again', 'red', self.try_again_register_new_user)
        self.try_again_button_register_new_user_window.place(x=750, y=400)

        self.capture_label = util.get_img_label(self.register_new_user_window)
        self.capture_label.place(x=10, y=0, width=700, height=500)

        #Function to add image to label
        self.add_img_to_label(self.capture_label)
        #Textbox to add username
        self.entry_text_register_new_user = util.get_entry_text(self.register_new_user_window)
        self.entry_text_register_new_user.place(x=750, y=150)
        #Add the text label to the textbox
        self.text_label_register_new_user = util.get_text_label(self.register_new_user_window, 'Please, \ninput username:')
        self.text_label_register_new_user.place(x=750, y=70)

    def try_again_register_new_user(self):
        #Exit the second window and get back to main window
        self.register_new_user_window.destroy()

    def add_img_to_label(self, label):
        imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
        label.imgtk = imgtk
        label.configure(image=imgtk)
        #To get the most recent image to be registered in db
        self.register_new_user_capture = self.most_recent_capture_arr.copy()

    def start(self):
        #Start the application
        self.main_window.mainloop()
    #Add data to database on clicking the accept button
    def accept_register_new_user(self):
        #get name from entry field
        name = self.entry_text_register_new_user.get(1.0, "end-1c")
        # Save image in directory
        cv2.imwrite(os.path.join(self.db_dir, '{}.jpg'.format(name)), self.register_new_user_capture)

        # #Get face embeddings from image
        # embeddings = face_recognition.face_encodings(self.register_new_user_capture)[0]
        #
        # file = open(os.path.join(self.db_dir, '{}.pickle'.format(name)), 'wb')
        # pickle.dump(embeddings, file)
        #Show alert that registration wa ssuccessful
        util.msg_box('Success!', 'User was registered successfully !')
       #Destroy the register user window
        self.register_new_user_window.destroy()


if __name__ == "__main__":
    app = App()
    app.start()

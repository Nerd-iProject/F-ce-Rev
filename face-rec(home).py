

### IMPORTING THE NECESSARY LIBRARIES

from flask import Flask, flash, request, redirect, url_for, render_template, send_from_directory
import urllib.request
import os
from werkzeug.utils import secure_filename

__author__='souhardya'


#Flask constructor takes the name of current module (__name__) as argument
app = Flask(__name__)


APP_ROOT = os.path.dirname(os.path.abspath(__file__))


#this are the roots what happens if the user navigate to these roots
#"/" is the basic route of the local host
#render_template is used for the redirecting to the html file
#route() function is a decorator, which tells the application which URL should call the associated function.
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/home",methods=['POST'])
def upload():
    target=os.path.join(APP_ROOT,'static/')
    print(target)

    ## the if condition will make a images folder is the folder is not present
    if not os.path.isdir(target):
        os.mkdir(target)

    #we have allowed multiple files for upload so files.getlist
    for file in request.files.getlist("file"):

        #file is coming here as an object
        print(file)
        filename=file.filename
        destination="/".join([target,"temp.jpg"])
        print(destination)
        file.save(destination)
    return render_template("complete.html")




@app.route("/<name>")
def user(name):
    return f"Hello {name}"

@app.route("/admin")
def admin():
    return redirect(url_for("home"))

if __name__=="__main__":
    app.run()

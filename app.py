'''The Flask application:
serves the HTML, serves static JS/CSS ,endpoints for saving trajectories or potentials (optional)
It does not run the simulation.'''

from flask import Flask, render_template


app=Flask(__name__)



if __name__=='__main__':
    app.run(debug=True)

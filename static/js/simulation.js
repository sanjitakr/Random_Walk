/* This is the core of the simulation.
Split into clean, modular physics + rendering components.

simulation.js

Handles:
main animation loop (requestAnimationFrame)
timestep integration

calls particle.update()

calls renderer.draw()

This is the “brain” that orchestrates everything.*/
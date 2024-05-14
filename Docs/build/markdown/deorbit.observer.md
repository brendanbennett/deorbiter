# deorbit.observer package

## Submodules

## deorbit.observer.observer module

### *class* deorbit.observer.observer.Observer(\*\*kwargs)

Bases: `object`

#### plot_config()

Method which gives a visual representation of radar station layout

#### run(sim_states, sim_times, checking_interval)

Runs the simulation of the radar measurements using an instance of the deorbit simulation.
Radars check for line of sight at regular time intervals equal to (checking_interval \* simulator interval) seconds.
Returns self.observed_times and self.observed_states class attributes containing the times and states
when the satellite has been observed by a radar station.

## Module contents

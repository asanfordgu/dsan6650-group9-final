
# **Emergency-Aware Reinforcement Learning for Urban Traffic Signal Control Using Real PeMS and EMS Dispatch Data**

## **Abstract**

Emergency response depends on an ambulance’s ability to travel quickly through heavy traffic. Congested streets in San Francisco slow ambulances because standard signal control does not adjust quickly enough when necessary, thereby impeding emergency responses. The current study trains a reinforcement learning traffic signal controller that uses three real-world data sources: California PeMS District 4 traffic detector records, their metadata, and San Francisco EMS 911 dispatch logs. These data inputs support a SUMO setup that models real congestion, emergency call timing, and ambulance routing. A Proximal Policy Optimization (PPO) agent is therefore trained to reduce delay for emergency vehicles and tested against fixed-time control,  and greedy preemption. Across repeated simulations, the RL controller outperformed fixed-time and greedy baselines on key mobility metrics, indicating that data-driven policies can adapt more effectively to real congestion patterns. These results suggest that reinforcement learning offers a promising alternative to traditional emergency preemption systems in dense urban networks.

## **Data**

* **PeMS 5-minute traffic records**: flow, speed, occupancy
* **PeMS metadata**: station locations, freeway IDs, lane counts
* **SF EMS dispatch logs**: timestamps, priorities, incident coordinates


## **Methods**

* SUMO network built from OpenStreetMap
* RL environment implemented with Gymnasium + TraCI
* PPO (Stable-Baselines3) as the main controller
* Baselines:

  * **Fixed-time**
  * **Greedy preemption**

## **Results**

Across repeated simulations, the PPO controller reduced emergency vehicle delay and produced smoother overall traffic compared to both baselines. The results show that RL can respond to real congestion patterns better than static or overly aggressive preemption strategies.

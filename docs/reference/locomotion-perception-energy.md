# Locomotion vs Perception Energy demands

The perception, guidance, navigation, and control (PGNC) systems of autonomous robots, which are often heavily reliant on power-hungry sensors and processors, are becoming a significant part of their total energy consumption. While locomotion remains the largest consumer of energy for most mobile robots, the power demands of AI-based perception are increasingly limiting operational time, especially in smaller form factors with strict weight and energy constraints.

The table below outlines the approximate breakdown of energy consumption and typical battery capacity for various robotic form factors. These values are highly dependent on the specific model, operational environment, and mission.

| Robot Form Factor | Locomotion Energy % | PGNC Energy % | Typical Battery Capacity |
| :--- | :---: | :---: | :--- |
| **Electric Truck** | 90-95% | 5-10% | 500-1,000 kWh |
| **Robotaxi** | 85-90% | 10-15% | 35-100 kWh |
| **Personal EV** | 90-95% | < 5% | 40-100 kWh |
| **Warehouse AMR** | 50-60% | 40-50% | 20-50 Ah (24V) |
| **Humanoid Robot** | 60-70% | 30-40% | 0.8-1.5 kWh |
| **Winged Drone** | 95-99% | < 5% | 10-30 Ah (22.2V) |
| **Quadcopter** | 80-90% | 10-20% | 1-2 Ah (11.1V) |

***

### Electric Trucks, Robotaxis, and Personal EVs 🚚🚗🚕

For large ground vehicles, **locomotion is the overwhelming driver of energy consumption**. The energy required to move a multi-ton vehicle is orders of magnitude greater than the power needed to run its on-board computers and sensors. For personal electric vehicles (EVs) with basic ADAS (Advanced Driver-Assistance Systems), the power draw of the PGNC systems is almost negligible, typically less than 5%. In contrast, autonomous electric trucks and robotaxis have much more powerful on-board computing, with power consumption sometimes reaching over 1 kW. Despite this, it's still a small fraction compared to the tens of kilowatts needed for propulsion. For example, a vehicle driving on the highway at a steady speed might use 20-30 kW for locomotion, while its autonomous hardware draws 1-2 kW.

***

### Warehouse AMRs and Humanoid Robots 🤖🦾

In smaller, lighter-weight robots, the energy consumption of PGNC systems becomes a much larger proportion of the total. **Warehouse Autonomous Mobile Robots (AMRs)** and **humanoid robots** operate in complex, dynamic environments that require constant perception and decision-making. The motors for locomotion are less powerful, and the time spent on perception, planning, and communication is relatively high.

 * A warehouse AMR might spend a significant amount of its operational time idling while it plans a path, waits for a human to pass, or navigates a cluttered space. During this time, the PGNC systems continue to consume power.

 * Similarly, a **humanoid robot's** power consumption is a finely balanced act. Its locomotion is incredibly inefficient compared to wheeled robots, with actuators and joints constantly drawing power to maintain balance and posture. However, its sophisticated perception, powered by multiple cameras, lidars, and powerful processors, can account for a significant portion of the total energy, often 30-40%.

***

### Drones and Quadcopters 🚁🪰

Flying robots have the most severe energy constraints. For both **winged drones** and **quadcopters**, the energy required for locomotion—specifically, to generate lift and overcome drag—is by far the dominant factor.

* A **quadcopter** must use all four propellers constantly to stay airborne (hovering). This means that even when stationary, its motors are drawing significant power, making its locomotion-to-PGNC ratio still high. The on-board computer for control and perception, while a small percentage of the total, is often the bottleneck for flight time.

* In contrast, a **winged drone** is much more energy-efficient in forward flight, as the wings generate lift passively. This dramatically reduces the locomotion energy once it's airborne, making the percentage for PGNC systems more significant than one might initially think, but still a minority of the total power draw. This is why even a small increase in the weight of a PGNC system can drastically reduce a drone's flight time.

## Mission Time

A robot's **mission time**, or operational duration, is a critical factor influenced by the balance between energy consumption for locomotion and for perception, guidance, navigation, and control (PGNC). For smaller, more agile robots like drones and humanoids, the energy demands of PGNC systems can significantly limit their useful operational time, leading to frequent recharging. For large ground vehicles, locomotion is the primary energy sink, and mission time is often limited by charging infrastructure rather than on-board processing.

Here's an updated table with typical mission times for various robot form factors:

| Robot Form Factor | Locomotion Energy % | PGNC Energy % | Typical Battery Capacity | Typical Mission Time |
| :--- | :---: | :---: | :--- | :--- |
| **Electric Truck** | 90-95% | 5-10% | 500-1,000 kWh | 6-12 hours |
| **Robotaxi** | 85-90% | 10-15% | 35-100 kWh | 4-8 hours |
| **Personal EV** | 90-95% | < 5% | 40-100 kWh | 4-10 hours |
| **Warehouse AMR** | 50-60% | 40-50% | 20-50 Ah (24V) 0.45-1.1 kWh | 6-8 hours |
| **Humanoid Robot** | 60-70% | 30-40% | 0.8-1.5 kWh | 1-2 hours |
| **Winged Drone** | 95-99% | < 5% | 10-30 Ah (22V) 0.22-0.67 kWh | 1-3 hours |
| **Quadcopter** | 80-90% | 10-20% | 1-2 Ah (11.1V) | 15-30 minutes |

A 10-30Ah battery capacity for a winged drone, which typically operates at 22.2V, translates to a range of 0.22 kWh to 0.67 kWh. 

A 20-50Ah battery capacity for an AMR, operating at 24V, translates to a range of 0.45 kWh to 1.1 kWh.

***

### Ground Vehicles: Electric Trucks, Robotaxis, and Personal EVs

The mission time of large ground vehicles is defined more by their overall range and the availability of charging infrastructure than by their PGNC power demands.

**Electric trucks** and **personal EVs** are built for long-distance travel, and their operation is typically constrained by the distance they can travel on a single charge. A driver might operate a truck for a full shift before needing a multi-hour charging stop or a battery swap. For a **robotaxi**, a full 24/7 operation is often achieved through a fleet management system where vehicles are routed to a charging station several times a day. While their PGNC systems consume power, the primary energy drain is still from moving the vehicle's massive weight and payload, making charging time the key bottleneck for operational availability.

***

### Industrial & Humanoid Robots

For indoor robots, mission time is directly tied to the efficiency of their locomotion and the power-hungry nature of their perception systems. **Warehouse AMRs** are designed for continuous operation in a structured environment. They can often "opportunistically charge" by returning to a charging station during periods of low activity or while waiting for their next task, allowing them to operate for a full work shift (6-8 hours) without a long break.

In stark contrast, **humanoid robots** have a significantly shorter mission time. Their bipedal locomotion is inherently inefficient, requiring constant energy expenditure to maintain balance and posture, in addition to movement. This, combined with the power demands of high-end sensors and processors for complex tasks like object manipulation and human interaction, results in a battery life of just 1-2 hours before needing a recharge. This is a major limiting factor for their practical use in many applications.

***

### Aerial Drones

Drones face the most stringent energy constraints, with a direct trade-off between payload, flight time, and operational range. For a **quadcopter**, the continuous power required to hover means flight times are very short, typically under 30 minutes. Adding advanced perception sensors like lidar or high-resolution cameras drastically reduces this time further, often to just 15-20 minutes. A **winged drone**, on the other hand, is much more efficient in forward flight. By gliding, it can dramatically reduce its locomotion energy, allowing for much longer mission times of 1-3 hours, making it suitable for applications like long-range surveying or delivery. However, even for winged drones, the weight and power demands of perception systems remain a key limiting factor for their maximum flight time and payload capacity.

## Power Analysis

Understanding the current draw is key to quantifying how long an EV can operate. The calculation is straightforward and relies on a fundamental electrical formula.

### The Calculation

To determine the current (amps) drawn by an electric vehicle's motor, we use the basic power formula:

**Power (P) = Voltage (V) × Current (I)**

Rearranging this formula to solve for current gives us:

**Current (I) = Power (P) / Voltage (V)**

 * **Power (P)** is the energy consumption of the EV's motor, which you provided as 20-30 kW (or 20,000-30,000 watts) at highway speed.

 * **Voltage (V)** is the operating voltage of the EV's battery pack.

 * **Current (I)** is the amperage we need to find.

### Typical EV Design and Voltage

The vast majority of modern electric vehicles are designed with a **400-volt (V)** architecture. This voltage level is a result of connecting hundreds of individual battery cells in a series to achieve the necessary power output. For example, a single lithium-ion cell typically has a nominal voltage of 3.6V, so a 400V pack would require about 111 cells in series. Some newer, higher-end EVs are transitioning to an **800-volt (V)** architecture to enable faster charging and reduce the weight of wiring due to lower current requirements.

For our calculation, we'll use the more common **400V** standard.

### Example Calculation

Let's calculate the current draw for an EV consuming a typical **25 kW** of power at highway speed.

1.  **Convert power to watts**:

    $25 \text{ kW} = 25,000 \text{ W}$

2.  **Use the formula with a 400V system**:

    $I = \frac{25,000 \text{ W}}{400 \text{ V}} = 62.5 \text{ A}$

To maintain a speed that consumes 25 kW, the EV's battery would be supplying **62.5 amps**.

If we were to use an 800V architecture for the same power consumption, the current draw would be halved:

$I = \frac{25,000 \text{ W}}{800 \text{ V}} = 31.25 \text{ A}$

This demonstrates why higher-voltage systems are more efficient—they require less current to deliver the same power, which reduces heat loss in the cables and motors.

### Quantifying Mission Time

Knowing the current draw allows us to connect power consumption to battery capacity, which is measured in kilowatt-hours (kWh). A kilowatt-hour represents the amount of energy equivalent to using one kilowatt of power for one hour.

To find the mission time, you can use the formula:

**Mission Time (hours) = Battery Capacity (kWh) / Average Power Consumption (kW)**

For example, a personal EV with a **75 kWh battery** driving at a constant highway speed and consuming **25 kW** would have a mission time of:

$75 \text{ kWh} / 25 \text{ kW} = 3 \text{ hours}$

This is a simplified calculation that assumes no energy is lost to heat or other systems (like air conditioning, lights, or the PGNC systems). In reality, the actual operational time would be slightly less.

# Energy efficiency of EVs

That's an excellent question, and you've identified a key detail. The 20-30 kW figure I provided is a general range that is indeed on the higher end of what a typical passenger EV consumes to maintain highway speed, particularly for larger or less aerodynamic vehicles. It's more accurate to think of it as a power *demand* for a specific speed, not a constant consumption for all EVs.

Here’s a more detailed breakdown to clarify the energy consumption of EVs:

### The Dominant Factor: Aerodynamic Drag

Unlike gasoline cars, which are most efficient at highway speeds due to engine design, EVs become **less efficient at higher speeds**. This is because the primary force an EV must overcome to maintain speed is **aerodynamic drag**, which increases with the square of the vehicle's speed. At 70 mph, a significant majority of the energy is being spent simply pushing air out of the way.

For a mid-sized sedan like a Tesla Model 3 or a Hyundai Ioniq 6, the continuous power required to maintain 70 mph on a flat road is typically in the range of **15-20 kW**. For a larger, less aerodynamic vehicle like an electric SUV or truck, this figure can easily rise to 25-30 kW or even higher. So while 25 kW is on the high side for an average EV, it's a realistic figure for a larger vehicle or for a car driving at a higher speed (e.g., 75-80 mph).

### Energy Efficiency Breakdown: City vs. Highway

The energy efficiency of an EV is typically measured in **miles per kilowatt-hour (miles/kWh)**, which is the inverse of the power-based calculation we did. The higher the miles/kWh, the more efficient the vehicle.

* **City Driving**: This is where EVs truly shine. In stop-and-go traffic, an EV's **regenerative braking** system captures a significant amount of the kinetic energy that would otherwise be lost as heat in the brakes. This energy is converted back into electricity and returned to the battery. While a gasoline car is at its least efficient in city traffic, an EV can be incredibly efficient, with consumption figures often around **3.5 - 4.5 miles/kWh**.

* **Highway Driving**: At a constant high speed, the opportunity for regenerative braking is minimal or non-existent. The vehicle is constantly expending energy to overcome rolling resistance and, more importantly, aerodynamic drag. This results in a much lower energy efficiency. A typical EV's efficiency at highway speeds (e.g., 65-75 mph) drops to around **2.5 - 3.5 miles/kWh**.

| Driving Condition | Efficiency (miles/kWh) | Reason for Efficiency |
| :--- | :---: | :--- |
| **City / Urban** | 3.5 - 4.5 | High use of regenerative braking and lower speeds. |
| **Highway** | 2.5 - 3.5 | Minimal regenerative braking and high aerodynamic drag. |

### Real-World Example

Let's use a more conservative highway consumption figure of **18 kW** for a mid-sized sedan at 70 mph and a typical **75 kWh battery** to recalculate the mission time.

1.  **Power (P)** = 18 kW

2.  **Battery Capacity** = 75 kWh

$Mission\ Time\ (hours) = \frac{75 \text{ kWh}}{18 \text{ kW}} \approx 4.17 \text{ hours}$

This is a more realistic figure for a highway trip than the 3 hours we calculated before with a higher power consumption. This highlights how a seemingly small change in power consumption can have a major impact on total mission time.

In summary, the 20-30 kW figure is a broad range that reflects the wide variety of electric vehicle sizes and driving conditions. The key takeaway is that an EV's energy consumption is heavily dependent on speed and driving style, with efficiency dropping off significantly on the highway due to aerodynamic drag.

## Comparing energy efficiency of EV vs Combustion

You can compare EVs to combustion engine vehicles in terms of energy efficiency by using a normalized metric called **miles per gallon gasoline equivalent (MPGe)**. The MPGe rating is an EPA standard designed to level the playing field by converting the energy in a gallon of gasoline to its electrical equivalent.

### Comparing Energy Efficiency

Electric vehicles are vastly more energy-efficient than internal combustion engine (ICE) vehicles.

* **Electric Vehicles (EVs)**: An EV motor is remarkably efficient, converting over **77%** of the electrical energy from the grid into power at the wheels. This high efficiency is due to the direct conversion of electrical energy to mechanical motion with minimal heat loss. Regenerative braking further boosts efficiency by recovering kinetic energy during deceleration, converting it back into electricity to recharge the battery.

* **Combustion Engine Vehicles**: An ICE is highly inefficient, converting only about **12% to 30%** of the energy stored in gasoline into useful power at the wheels. Most of the remaining energy is lost as waste heat through the exhaust and engine block, especially during city driving and idling.

### Normalizing Efficiency with MPGe

To normalize efficiency and directly compare the two, the U.S. Environmental Protection Agency (EPA) established the MPGe metric. The standard conversion is based on the energy content of one gallon of gasoline, which is approximately **33.7 kilowatt-hours (kWh)** of electricity.

To calculate the MPGe for an EV, you take the vehicle's efficiency in miles per kWh and multiply it by this conversion factor:

$MPGe = \text{Vehicle Efficiency (miles/kWh)} \times 33.7 \text{ kWh/gallon}$

For example, a typical EV with a highway efficiency of 3.0 miles/kWh would have an MPGe rating of:

$3.0 \times 33.7 = 101.1 \text{ MPGe}$

This simple calculation shows that an EV with an efficiency of 3 miles/kWh is roughly equivalent to a gasoline car that gets over 100 miles per gallon. 

***

### Cost Comparison

The cost of energy is the final piece of the puzzle, and it's heavily dependent on regional prices. The table below shows a comparison of the typical cost per mile for both EVs and gasoline cars, based on national averages.

| Vehicle Type | Average Efficiency | Normalized Cost Unit | Average Cost of Energy | Typical Cost per Mile |
| :--- | :---: | :---: | :--- | :---: |
| **Gasoline Car** | 25 MPG | Gallon | $3.50/gallon | **$0.14** |
| **EV (Home Charging)** | 3.5 miles/kWh | kWh | $0.15/kWh | **$0.04** |
| **EV (Public Fast Charging)** | 3.5 miles/kWh | kWh | $0.45/kWh | **$0.13** |

* **Gasoline Cars**: The cost per mile is calculated by dividing the average gasoline price by the vehicle's MPG. For a car that gets 25 MPG with gasoline at $3.50/gallon, the cost is $3.50 / 25 miles = **$0.14 per mile**.

* **Electric Vehicles**: The cost per mile for an EV is determined by multiplying the vehicle's energy consumption (kWh/mile) by the cost of electricity ($/kWh). This cost varies significantly based on where and when you charge.

    * **Home Charging** 🏡: For most EV owners, home charging is the most common and cheapest method. The average residential electricity rate in the US is about $0.15/kWh, leading to a cost per mile of roughly **$0.04**.

    * **Public Fast Charging** ⚡: Fast chargers are more expensive and are used primarily for long road trips. Prices can range from $0.30 to over $0.60 per kWh. Using a high-end average of $0.45/kWh, the cost per mile rises to about **$0.13**, making it comparable to or slightly cheaper than a gasoline car.

Overall, while EVs have a higher initial purchase price, their significantly lower cost of energy, coupled with reduced maintenance needs, often makes them cheaper to own and operate in the long term.

## Energy efficiency of small cars and hybrids

The previous figure of 25 MPG is a broad average that doesn't fully capture the efficiency of modern, smaller cars and especially hybrids. It's a national average that includes all vehicle types, including large SUVs and trucks, which significantly pull that number down.

Let's re-evaluate the comparison with modern small cars and hybrids.

### The Fundamental Efficiency Difference

Even with the impressive advancements in gasoline and hybrid technology, the core principle of energy efficiency remains the same: an electric motor is fundamentally more efficient than an internal combustion engine (ICE).

* **Electric Motor**: A typical EV motor converts **over 80%** of the electrical energy from the battery into motion. This is a very direct and efficient process.

* **Combustion Engine**: A modern ICE, even a highly optimized one, only converts **20-30%** of the energy in gasoline into motion. The rest is lost as waste heat through the exhaust and cooling systems.

This fundamental inefficiency is why even a hybrid vehicle, which combines an ICE with an electric motor, cannot match the overall energy efficiency of a full EV.

### The Role of Regenerative Braking: Hybrid vs. EV

Just like EVs, hybrids use regenerative braking. However, the system's function and scale are fundamentally different from an EV's.

* **Hybrid**: A hybrid uses a small electric motor and battery to capture kinetic energy during braking, which helps the gasoline engine run more efficiently and provides a power boost. The energy captured is a significant help, but it's a small-scale system designed to improve MPG, not as the primary source of propulsion.

* **EV**: An EV's regenerative braking is a much more powerful, large-scale system. It is the primary means of slowing down the vehicle, and it can recapture a much larger percentage of the braking energy. In many cases, it allows for "one-pedal driving," where lifting off the accelerator slows the car down enough to almost completely negate the need for the friction brakes in normal driving.

### Normalizing Efficiency with Modern Vehicles

Let's redo the MPGe comparison using a popular, modern gasoline car and a top-selling hybrid.

* **Modern Gasoline Car** (e.g., a 2024 Toyota Corolla): This vehicle achieves a fantastic **35 MPG combined**.

* **Modern Hybrid** (e.g., a 2024 Toyota Prius): This vehicle takes advantage of regenerative braking to achieve an impressive **57 MPG combined**.

* **Modern EV** (e.g., a 2024 Tesla Model 3): This vehicle is rated by the EPA at **132 MPGe combined**.

### Updated Cost Comparison

Using the same energy prices from before, let's see how the cost-per-mile changes with these more efficient vehicles.

| Vehicle Type | Average Efficiency | Normalized Cost Unit | Average Cost of Energy | Typical Cost per Mile |
| :--- | :---: | :---: | :--- | :---: |
| **Modern Gasoline** | 35 MPG | Gallon | $3.50/gallon | **$0.10** |
| **Modern Hybrid** | 57 MPG | Gallon | $3.50/gallon | **$0.06** |
| **EV (Home Charging)** | 3.5 miles/kWh (132 MPGe) | kWh | $0.15/kWh | **$0.04** |
| **EV (Public Fast Charging)** | 3.5 miles/kWh (132 MPGe) | kWh | $0.45/kWh | **$0.13** |

### Conclusion

While a modern hybrid is a great leap forward in efficiency and cost savings compared to older gasoline cars, an EV still holds a substantial advantage. It simply takes far less energy to propel an electric car a given distance than a gasoline-powered one, regardless of how advanced the ICE may be. The high-efficiency of an EV's motor, combined with its superior regenerative braking, gives it a fundamental advantage that an internal combustion engine cannot overcome.

## Energy Efficiency of Electric vs Combustion

A hybrid can, in some specific scenarios, be more **cost-efficient** than an EV. However, that does not mean it is more **energy-efficient**. 

### 1. Energy Efficiency (The Physics)

This is a measure of how well a vehicle converts a unit of energy into motion. This is the metric that the **MPGe** standard was designed for. It is a pure scientific comparison that ignores the price of the energy.

* **Hybrid Vehicle (Toyota Prius):** The Prius achieves its high MPG (57 MPG) by a complex combination of a gasoline engine and an electric motor. But remember, the gasoline engine itself is only 20-30% efficient. To get to 57 MPG, it has to use a gallon of gasoline's worth of energy (33.7 kWh), and most of that energy is still lost as heat.

* **Electric Vehicle (Tesla Model 3):** An EV's electric motor is over 80% efficient. The vast majority of the electrical energy is converted directly into motion. This is why the EPA rates it at **132 MPGe**.

When you compare 57 MPGe to 132 MPGe, the conclusion holds: **the EV is more than twice as energy-efficient as the hybrid.** It takes far less energy, regardless of its cost, to move the EV one mile.

### 2. Cost Efficiency (The Economics)

This is a measure of how much it costs you to drive a certain distance. This is where the price of the fuel (gasoline vs. electricity) comes into play.

The public fast charging network is significantly more expensive than the average cost of gasoline. So for long road trips, hybrids are more cost-effective.

The reality is that **90% of EV charging happens at home**, where the cost is dramatically lower. Public fast charging is a premium, on-the-go service, priced higher to cover the cost of the infrastructure and the speed of the charge.

Let's use the average national prices to create a more realistic and complete picture:

| Vehicle Type | Average Energy Efficiency | Cost of Energy | Cost Per Mile |
| :--- | :---: | :---: | :---: |
| **Hybrid** | 57 MPG | $3.50 / Gallon | **$0.06** |
| **EV (Home Charging)** | 3.5 miles/kWh (132 MPGe) | $0.17 / kWh | **$0.05** |
| **EV (Public Fast Charging)**| 3.5 miles/kWh (132 MPGe) | $0.45 / kWh | **$0.13** |

As this table shows, if you compare the most common use case for an EV (home charging) with a hybrid, **the EV is still more cost-efficient**. It's only when you rely heavily on public fast charging that the cost per mile can become comparable to, or even more expensive than, a gasoline or hybrid car.

The EV's significant advantage in **energy efficiency** (MPGe) generally translates to a lower **cost** per mile for the vast majority of drivers who charge at home. The single scenario where an EV's cost per mile is higher is due to the business model and convenience premium of public fast charging.

## Cold weather efficiency

The EPA's MPGe rating doesn't account for real-world scenarios like extreme temperatures. Cold weather significantly impacts an EV's range and an internal combustion engine (ICE) has a unique advantage by using its waste heat for cabin heating.

Here is a calculation of a combustion engine's efficiency when you account for the energy used for heating.

### The Calculation for Combustion Engine Efficiency

To understand the full picture, we have to look at how much energy is in a gallon of gasoline and how that energy is distributed.

* **Total Energy in one Gallon of Gasoline:** A gallon of gasoline contains approximately **33.7 kWh** of energy.

* **Engine Efficiency:** As we've discussed, an ICE is only about 20-30% efficient at converting this energy into motion. This means for every 33.7 kWh, roughly **10.1 kWh** becomes useful work (propulsion).

* **Waste Heat:** The remaining energy, about **60-70%**, is lost as waste heat. This equates to over **20 kWh** of waste heat per gallon, which is where the cabin heat comes from.

### Quantifying the Heater's Impact

The heat for a combustion car's cabin is a byproduct of the engine running. The car's heating system simply circulates the hot coolant through a heater core and blows air over it to warm the cabin. The fan that blows the air is electric, but its power draw is minimal—typically less than 1 kW. The main energy for heating is already "free," as it's a byproduct of the engine's inefficiency.

* **Heater Power Consumption:** A typical car heater draws very little power. The fan may use a few hundred watts, and the energy comes from the car's alternator, which is powered by the engine. The impact on fuel consumption is minimal.

### The EV Heater's Energy Consumption

For an EV, the heater is a major energy drain because it has to generate heat from scratch using the battery's energy.

* **Resistive Heaters:** Older EVs use simple resistive heaters, essentially giant space heaters, which can draw anywhere from **3-5 kW** of power. This is why you see a dramatic drop in range.

* **Heat Pumps:** Modern EVs, like the Tesla Model 3, use heat pumps, which are much more efficient. They work like a reverse air conditioner, moving heat from the outside air into the cabin. This is much more efficient, but it's still a significant power draw, often **1-2 kW** on a cold day. 

***

### The Final Comparison

Even when including the effect of cold-weather heating, an EV remains more **energy-efficient** than an ICE vehicle. This is because the fundamental inefficiency of the gasoline engine (losing 60-70% of energy as heat) is a constant factor. While the EV loses range by using its battery for heating, the ICE car wast heat reuse is not sufficient to overcome the inefficiency of the combustion process.

## Performance Comparison

The typical power output of a single electric motor in a modern EV can range from 50 kW for a small, entry-level car to 300 kW or more for a high-performance model. It's important to understand the distinction between continuous power and peak power.

Continuous vs. Peak Power

* Continuous Power: This is the power an electric motor can sustain for an extended period without overheating. For most EVs, the continuous power rating is significantly lower than the peak power, as it represents the steady-state performance needed for normal driving.

* Peak Power: This is the maximum power an electric motor can deliver for a short burst, such as during rapid acceleration or when climbing a steep hill. This can be more than double the continuous power rating and is often the number manufacturers use for marketing because it's the more impressive figure.

### Typical Power Ranges for EVs

The power output of an EV's motor is analogous to a combustion engine's horsepower, with 1 kW roughly equal to 1.34 horsepower.

* Compact EVs: Motors in smaller EVs typically produce between 50 kW and 100 kW. This is sufficient for city driving and provides adequate acceleration for highway use.

* Mid-size to Large EVs: Most mid-range EVs have motors in the 150 kW to 250 kW range. Many of these vehicles have multiple motors (one for each axle) to provide all-wheel drive, with the total system power being the sum of the individual motors.

* High-Performance EVs: High-end models from companies like Tesla and Porsche can have motors that produce well over 300 kW each. When paired with multiple motors, the total system output can exceed 700 kW, leading to exceptional acceleration and top speeds.

### Typical Power Ranges for Combustion Engines

The performance range of combustion engines in kilowatts (kW) is very wide, but a general equivalent to horsepower (HP) is based on the conversion factor that **1 HP = 0.746 kW**.

To convert a horsepower rating to kilowatts, you simply multiply the HP by 0.746. For example, a 150 HP engine is equal to 111.9 kW ($150 \\times 0.746 = 111.9$).

Here's the typical range for different types of internal combustion engine vehicles:

  * **Small, Economy Cars:** These engines typically have a power output between **50 to 100 kW** (67 to 134 HP). This is enough for efficient city driving and adequate performance for daily commuting.

  * **Family Sedans and SUVs:** The average family vehicle has an engine that produces between **110 to 200 kW** (147 to 268 HP). This power range provides a good balance of performance for highway cruising and city driving.

  * **High-Performance and Sports Cars:** Engines in this category can easily exceed **220 kW**, with many producing over **370 kW** (500 HP) or more. These engines are designed for maximum acceleration and top speed.








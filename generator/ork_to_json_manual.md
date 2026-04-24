# TRANSLATING ORK TO JSON INSTRUCTION
## REQUIERMENTS:
 RocketSerializer packet from pip
## STEP 1:
Paste your OpenRocket.jar file into your work directory
##### WARNING:
Only OpenRocket 23 will work
## STEP 2: 
Open file /venv/lib/python3.14/site-packages/rocketserializer/components/motor.py

(name of directory may differ)

## SETP 3:
Comment folowing code:

`motor_length = float(bs.find("motormount").find("length").text)
`

`motor_radius = float(bs.find("motormount")find("diameter").text) 
`  

After that, you need to define motor_length and motor_radius. For example:

`
motor_length = 0.314
`

`
motor_radius = 0.054/2
`
## STEP 4:
execute 
`
ork2json -filename <filename>
`


## PS
If you wonder why u need to make all this flip flops to make this work.

Our .ork file was created in OpenRocket 24, and ork2json does not provide any support for any OpenRocket version beyond 23th

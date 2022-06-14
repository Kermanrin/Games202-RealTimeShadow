# Games202-RealTimeShadow
Final Project of CS238 SJTU

HOW TO RUN IT:

In this project, four parts of work were implemented :PCF, PCSS, Shadow map, Multiple source lights and dynamic object.

You should test the algorithm you want by annotated the rest two algorithms in the file named `phongFragment.glsl`

If you want to test the Multiple source light and dynamic object, you should enter the file named `engine.js` and turn the variable moving from `False` to `True`.

SOME INNOVATION:

1,Shadow Map: Mainly in the self shading .Provide a bias which is oriented to the angle between lightpos and normal and add a upper limit to avoid the (tan90)making rander strange.

2,PCF and PCSS: Add bias1 to avoid the self shading, add bias2 to avoid the noisy points on the platform.Some more implements for dicrease the noisy points(on the person model and the shadow) were done ,such as improving the projection window and 
the adjustment of the step in PCSS.

3,Dynamic object and Multiple source lights: Two more source lights were added ,and the loop of the track is a round.Some emm problems of loading the models  happpens sometimes with unkonw reason.


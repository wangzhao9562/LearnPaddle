# Boston House Pride Prediction    

## Implementation with manual network    
load_data : Load house pride data and generate appropriate datasets for network model  

Network - __init__ : Initialize weight and intercept through random way  
        - forward : Make initial prediction   
        - loss : Loss function  
		- gradient : Compute gradient of weight and intercepts  
		- update : Update weight and intercepts through SGD method  
		- train : Train network model   
  		
Train model  
  		
## Implementation with Paddle platform  
(import paddle libs : paddle, paddle.fluid, paddle.fluid.dygraph, paddle.fluid.dygraph.Linear)  
  
load_data ：Load house pride data and generate appropriate datasets for network model  
  
Regressor - __init__ : Inherite from fluid.dygraph.Layer, define a linear safe full-connected layer  
          - forward : Make initial prediction   
  		  
define environment for Paddle dynamic graph through 'with' in python - Train model  
                                                                     - load data  
																	 - Set SGD optimizer  
  																	 
Train model in paddle dynamic graph environment  


		

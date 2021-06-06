# Sparse_Signals_and_Denoising
Here I attempted to denoise a sparse signal that is corrupted by random noise.

Firstly I have created a 1x128 vector, x, with 5 non-zero coefficients as stated in the homework and added Gaussian noise to obtain the vector y. The aim of this part to solve the following equation: 

![image](https://user-images.githubusercontent.com/48417171/120928032-b9197680-c6eb-11eb-9c8a-4048bf987a46.png)

I solved this equation by using the fact that x_i, for 1≤i≤128 are independent, as follows: 

![image](https://user-images.githubusercontent.com/48417171/120928077-e23a0700-c6eb-11eb-87de-a38e374267c8.png)

These equations have shown that the best x ̂_i should be chosen as following:

![image](https://user-images.githubusercontent.com/48417171/120928090-eb2ad880-c6eb-11eb-9a91-ddcd5381c071.png)

For this algorithm I have find the best x vector for  λ=[0.01, 0.05, 0.1, 0.2], then I have plotted the error, L2 norm between actual sparse input and the found x vector. The result is as following. 
 
![image](https://user-images.githubusercontent.com/48417171/120928097-f41baa00-c6eb-11eb-8598-40103907e297.png)

From the selection criteria of the best x ̂_i we can see that when y_i is large compared to λ, x ̂_i takes the value of y_i-λ. Also, when the y_i is smaller compared to -λ then x ̂_i takes the value of y_i+λ. For other values, the x ̂_i becomes zero. This implies that if the absolute of the y_i is not large than λ the optimization will assume that the importance of the corresponding x_i is not much and x_i will be optimized to zero which makes the vector sparse. Therefore, increasing lambda value will result in sparser vector where the lambda acts as a threshold of the decision that determines when the x_i should be zero. 

Furthermore, from the Figure 1 we can see that as the lambda values increases first the error decreases and then it gets bigger again. This is due to the fact that our original x vector is sparse and as we increase the lambda the sparseness of the predicted x vector increases. Thus, the prediction gets sparser as the original x vector. However, increasing the lambda value a lot makes the error increase because this time optimization focuses so much on making predictions sparser that it starts to omit the first part of the error function which is decreasing the squared error between the predicted vector and the real vector x.

## Random Frequency Domain Sampling and Aliasing 

In this part first I have computed the Fourier transform of the sparse signal x. And then I have proceeded to compare equispaced and random undersampling. First, lets look at whether the  x_u has the Minimum Norm Least Squares solution, i.e.: 

![image](https://user-images.githubusercontent.com/48417171/120928120-07c71080-c6ec-11eb-92ec-07b0126d6fd2.png)

Here Fu^+ is the pseudoinverse of the matrix Fu. We know that the pseudoinverse gives the Minimum Norm Least Squares (MNLS) solution. So, I have done this calculation on code and plotted the x^* and xu on top of each other. The result is as following. 

![image](https://user-images.githubusercontent.com/48417171/120928128-0eee1e80-c6ec-11eb-9fd8-aca12c6fb7b9.png)

From this figure we can see that the xu and the MNLS solution has the same shape the only difference is in the magnitude. That is due to the fact that after taking the inverse Fourier we multiply the xu vector with 4 if we did not multiply it because we only have ¼ of the samples, without multiplying with 4 we get the same result as MNLS solution. To get the MNLS solution we solve the equation with Fu (under sampled Fourier matrix). This returns us the exact solution we get after taking the inverse of the Xu matrix. 

In other words, for sampling with the equal spaced frequency domain the best result we can get is xu, or mathematically since Fu=MF, where F and M (mixing matrix) are full column rank, Fu^+=F^+ M^+=F^(-1) M therefore the NMLS result, x^*=Fu^+ X=F^(-1) MX=F^(-1) Xu=ifftc(Xu)=xu.

Then I have proceeded with finding xr as stated in the homework and plotted x,xu,xr vectors individually and all together for better comparison purposes. The result is as following: 

![image](https://user-images.githubusercontent.com/48417171/120928147-20372b00-c6ec-11eb-9335-da0d0424568a.png)

The comprehensive sensing theory says that we can exactly recover sparse signals if we under sample, i.e., we can beat Nyquist Sampling Theory if we are trying to predict a sparse vector. Now we see in Figure 3 that because for the xu vector we uniformly down sampled we basically aliased higher frequencies and therefor the xu vector is sparse and is periodic. So, the sampling is inappropriate that information extracted to get the original x vector is not suitable. If we do random sampling, we are extracting lots of frequencies because the sample spacing is changing. Sometimes it is big sometimes it is not. However, this randomness also results like a noise like output, that is incoherent aliasing. The important thing we can see is that different from the xu vector xr has all the peeks that original vector x has but xr also has noise like components. Furthermore, these noise likely peeks are generally smaller than the peeks that correspond to the original vector x peeks, therefore if we find a way to eliminate these noisy like peeks, we can extract the original x vector. To do that we have added a L1 norm penalty which will eliminate the noisy like peeks and makes our predicted vector x a sparse vector just looks like the original vector x. 


** Reconstruction from Randomly Sampled Frequency Domain Data

The main purpose of this part is to solve the following equation:

![image](https://user-images.githubusercontent.com/48417171/120928171-36dd8200-c6ec-11eb-8a8d-9a41e33a6f6b.png)

Since the variables are coupled through Fourier transform, I solved this problem by utilizing ADMM. I have transformed the loss function above for ADMM and find the update rules as following:

![image](https://user-images.githubusercontent.com/48417171/120928195-4ceb4280-c6ec-11eb-8f70-a3bf798e045c.png)

With these update rules I have run the algorithm for the lambda values [0.01, 0.05, 0.1]. After some experiment I have selected rho to be equal to 10 where this value was giving roughly the best result. For the error metric I have chosen Mean Square Error and run the ADMM for 5000 iteration.

![image](https://user-images.githubusercontent.com/48417171/120928204-54125080-c6ec-11eb-9c3c-a8d5cb0cb6c2.png)

As we can see from the Figure 4 ADMM algorithm is able give us a good approximation of the real vector x where at the end the MSE error is 0.05. The first thing I have realized is that the algorithm converges much faster for the high Lambda values. I think that is because for high lambda values the indexes where the predicted array should have zero, converges to zero quicker, i.e., those values become zero more quickly because as we see in the part 1 this lambda values act like a threshold and high lambda value makes the vector sparser in a smaller number of iterations. Also, high lambda values make predicted vectors sparser but in the given lambda values the results do not differ much. However, I have done further experience and seen that for a higher lambda values the error gets even smaller. 

Finally, by adding L1 penalty term we can extract the original signal up to some point for lambda values [0.01, 0.05, 0.1]. The noisy like components are mostly gone and the predicted vectors look like the original x vector. The only problem is that on the original vector the lowest magnitude was 0.2 whereas the noisy like components was also may having magnitude slightly bigger than 0.2. Therefore, L1 norm extracted the original vector x only up to a point, as can be seen below.

![image](https://user-images.githubusercontent.com/48417171/120928215-5e344f00-c6ec-11eb-839e-9f14dc6bd655.png)



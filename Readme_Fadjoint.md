<div align="center">
 
# Backpropagation and $F$-adjoint
<div align="left">

# Main purpose
 
The main purpose of this project is to implement the well-known backpropagation algorithm in an easy manner based on the idea of the $F$-adjoint propagation proposed in the following arxiv preprint: "[Backpropagation and F-adjoint. arXiv preprint arXiv:2304.13820.](https://arxiv.org/abs/2304.13820)"

Hereafter, we shall recall the background and notation used in the above reference.
# Background and Notation 
We consider the simple case of a fully-connected deep multi-layer perceptron (MLP) composed of $L$ layers trained in a supervised setting. 
We will denote such an architecture by $A[N_0, \cdots, N_\ell,\cdots, N_L]$, where $N_0$ is the size of the input layer, $N_\ell$ is the size of hidden layer $\ell$,
and $N_L$ is the size of the output layer. In particular, we denote $W^{\ell}$ the Weight matrix of the layer ${\ell}$,  $W^{\ell}\in\mathbb{R}^{N_{\ell}\times N_{\ell-1}}$,
 $Y^{\ell}$ the preactivation vector at layer ${\ell}$, $Y^{\ell} = W^{\ell}X^{\ell-1}\in\mathbb{R}^{N_{\ell}}$, $X^{\ell}$ the activition vector at the layer ${\ell}$, $X^{\ell} =\sigma^{\ell}(Y^{\ell})\in\mathbb{R}^{N_{\ell}}$, where $\sigma^\ell$  is a fixed coordinate-wise activition function of the layer ${\ell}$, $\sigma^\ell :\mathbb{R}^{N_{\ell}}\ni Y^{\ell}\longmapsto\sigma^{\ell}(Y^{\ell})\in\mathbb{R}^{N_{\ell}}.$


# The $F$-propagation and $F$-adjoint
For the sake of coherency  of presentation we shall recall the definition of the this notion. 

## Definition of an $F$-propagation 

Let $X^0\in\mathbb{R}^{N_0}$ be a given data, $\sigma$ be a coordinate-wise map from $\mathbb{R}^{N_\ell}$ into $\mathbb{R}^{N_{\ell}}$ and $W^{\ell}\in \mathbb{R}^{{N_{\ell}}\times{N_{\ell-1}}}$ for all ${1\leq \ell\leq L}$. We say that we have a two-step recursive F-propagation   $F$  through the (MLP) $`A[N_0,\cdots, N_L]`$ if   one has the following family of vectors
```math
F(X^0):=\begin{Bmatrix}X^{0},Y^{1},X^{1},\cdots,X^{L-1},Y^{L},X^{L}\end{Bmatrix}\  \mathrm{where}\  Y^\ell=W^{\ell}X^{\ell-1}, \ X^\ell=\sigma(Y^\ell),\ \ell=1,\cdots, L.
```
Before going further, let us point that in the above definition the prefix "F" stands for "Feed-forward".

## Definition of the $F$-adjoint of an $F$-propagation

Let $X^0\in\mathbb{R}^{N_0}$ be a given data and let  $X^L_*\in\mathbb{R}^{N_L}$ be a given vector.  We define the F-adjoint propagation  $`{F}_{*}`$, through the (MLP) $`A[N_0,\cdots, N_L]`$, associated to the F-propagation  $`F(X^0)`$  as follows 
```math
F_{*}(X^{0}, X^{L}_{*}):=\begin{Bmatrix}X^{L}_{*}, Y^{L}_{*}, X^{L-1}_{*},\cdots, X^{1}_{*},Y^{1}_{*}, X^{0}_{*} \end{Bmatrix}\  \mathrm{where}\  Y^\ell_{*}=X^{\ell}_{*}\odot {\sigma}'(Y^\ell), \ X^{\ell-1}_{*}=(W^\ell)^\top Y^\ell_{*},\ \ell=L,\cdots, 1.
```
The following Key lemma provide the link between backpropagation and $F$-adjoint propagation. More precisely, we have a simple formulas to comput gradient with respect to the weight $`W^\ell`$.

## Lemma

For a  fixed data point  $`(x, y) \in \mathbb{R}^{N_0}\times\mathbb{R}^{N_L}`$, with feature vector $x$ and label $y$ and  a  fixed loss function $J$.  If $`X^{L}_{*}=\frac{\partial J}{\partial X^{L}}`$ then for any $`\ell\in\{1,\cdots, L\}`$, we have 
```math
 {Y_{*}^{\ell}}\left({X^{\ell -1}}\right)^\top=  \frac{\partial J}{\partial W^{\ell}}.
```
As a consequense, one may rewrite the backprpagation algorithm as follows:

1. Require: $`x,y`$
2. Ensure: $`W:=(W^\ell)_{1\leq \ell\leq L}`$ (Final weights)

   1.  Initialize $`W^\ell , \ell=1,\ldots,L`$
   2.  Function: $F$-propagation($x,W,\sigma$)
    
        1.  $`X^0\leftarrow x`$
        2.  $`F\leftarrow\{X^0\}`$
        3.  For $`\ell=1`$ to $L$:
                            
            $`Y^\ell\leftarrow W^\ell X^{\ell-1}`$
                 
            $`X^\ell\leftarrow\sigma(Y^{\ell})`$
            
            $`F\leftarrow Y^\ell,X^\ell`$
            
            End For
          
       Return $F$
    3.  Function: $F_*$-propagation($`F, J, y, \sigma',\eta`$)

        1. $`X_*^L\leftarrow \frac{\partial J}{\partial X^L}(X^L, y)`$        
        2. For $`\ell= L`$ to $1$:
                              
            $`Y_*^\ell\leftarrow X_*^{\ell}\odot\sigma'(Y^\ell)`$
           
           $`X^{\ell-1}_{*} \leftarrow \displaystyle({W^{\ell}})^\top Y^\ell_{*}`$
           
           $`W^\ell\leftarrow W^\ell-\eta Y_*^\ell(X^{\ell-1})^\top`$
           
           $`W\leftarrow W^\ell`$
           
           End For
           
     Return $W$
 

## Reference

<div id="refs" class="references">


Boughammoura, A. (2023). Backpropagation and F-adjoint. arXiv preprint arXiv:2304.13820.(https://arxiv.org/abs/2304.13820)

</div>

P.S.
Any comments and suggestions are welcome, and readers should feel free to contact me via the following e-mail: ahmadboughamoura@gmail.com


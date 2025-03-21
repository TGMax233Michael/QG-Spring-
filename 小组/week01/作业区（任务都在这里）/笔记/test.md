# 最小二乘法推导过程
## 向量求导定义
设 $f(\mathbf{x})$ 是关于 $\mathbf{{x}}$ 的函数，其中 $\mathbf{x}$ 是向量变元，并且
$\mathbf{x} = 
\begin{bmatrix}
    x_1 & x_2 & \cdots & a_n\\
\end{bmatrix}^ T
$ 

则
$$\frac{\partial f(\mathbf{x})}{\partial \mathbf{x}} =
\begin{bmatrix}
    \frac{\partial f(\mathbf{x})}{\partial x_1} & \frac{\partial fx\mathbf{x})}{\partial x_2} & \cdots & \frac{\partial f(\mathbf{x})}{\partial x_n}\\
\end{bmatrix}^T
$$

## 一些矩阵的求导法则
### 1. $\frac{\partial a}{\partial \mathbf{x}} = 0$
不作具体证明

### 2. $\frac{\partial \mathbf{A}^T\mathbf{x}}{\partial \mathbf{x}} = \mathbf{A}$
其中 
$$
\mathbf{A} = 
\begin{matrix}
    a_1 & a_2 & \cdots & a_n\\
\end{matrix}^T
$$

$$
\mathbf{x} = 
\left[
    \begin{matrix}
        x_1 & x_2 & \cdots & x_n\\
    \end{matrix}
\right]^T
$$

故  
  
$$f(\mathbf{x}) = a_1x_1 + a_2x_2 + \cdots + a_nx_n$$

对$f(X)$ 求 $x$ 的偏导, 则  

$$\frac{\partial \mathbf{x}^T\mathbf{A}}{\partial \mathbf{x}} = \frac{\partial \mathbf{A}^T\mathbf{x}}{\partial \mathbf{x}}$$
$$=\frac{\partial (a_1x_1 + a_2x_2 + \cdots + a_nx_n)}{\partial \mathbf(x)}$$
$$
=
\begin{bmatrix}
    \frac{\partial (a_1x_1 + a_2x_2 + \cdots + a_nx_n)}{x_1}\\
    \frac{\partial (a_1x_1 + a_2x_2 + \cdots + a_nx_n)}{x_2}\\
    \vdots\\
    \frac{\partial (a_1x_1 + a_2x_2 + \cdots + a_nx_n)}{x_n}\\
\end{bmatrix}^T
$$
$$
= 
\begin{bmatrix}
    a_1\\
    a_2\\ 
    \vdots\\
    a_n\\
\end{bmatrix}
 =\mathbf{A}
$$



### 3. $\frac{\partial (X^TX)}{\partial X} = 2X$
$$\frac{\partial (\mathbf{x}^T\mathbf{x})}{\partial \mathbf{x}}$$
$$=\frac{\partial (x_1^2+x_2^2+\cdots+x_n^2)}{\partial \mathbf{x}}$$
$$
=
\begin{bmatrix}
        \frac{\partial (x_1^2+x_2^2+\cdots+x_n^2)}{\partial x_1}\\
        \frac{\partial (x_1^2+x_2^2+\cdots+x_n^2)}{\partial x_2}\\
        \vdots\\
        \frac{\partial (x_1^2+x_2^2+\cdots+x_n^2)}{\partial x_n}\\
\end{bmatrix}
$$
$$
=
\begin{bmatrix}
    2x_1\\
    2x_2\\
    \vdots\\
    2x_n\\
\end{bmatrix}= 2\mathbf{x}
$$

### 4. $\frac{\partial X^TAX}{\partial X} = \mathbf{x}^T\mathbf{A}\mathbf{x}$
$$
\mathbf{x}^T\mathbf{A}\mathbf{x} = 
\begin{bmatrix}
  x_1&  x_2 & \cdots & x_n
\end{bmatrix}

\begin{bmatrix}
  a_{11} & a_{12}  & \cdots & a_{1n} \\
  a_{21} & a_{22}  & \cdots & a_{2n} \\
  \vdots & \vdots  & \ddots & \vdots \\
  a_{n1} & a_{n2}  & \cdots & a_{nn} \\
\end{bmatrix}

\begin{bmatrix}
  x_1\\
  x_2\\
  \vdots\\
  x_n\\
\end{bmatrix}
$$

$$
=
\begin{bmatrix}                                     
    a_{11}x_1+a_{21}x_2+\cdots+a_{n1}x_n & 
    a_{12}x_1+a_{22}x_2+\cdots+a_{n2}x_n & 
    \cdots & 
    a_{1n}x_1+a_{2n}x_2+\cdots+a_{nn}x_n
\end{bmatrix}

\begin{bmatrix}
  x_1\\
  x_2\\
  \vdots\\
  x_n\\
\end{bmatrix}
$$

$$
=
x_1(a_{11}x_1+a_{21}x_2+\cdots+a_{n1}x_n) + x_2(a_{12}x_1+a_{22}x_2+\cdots+a_{n2}x_n) +
\cdots + 
x_n(a_{1n}x_1+a_{2n}x_2+\cdots+a_{nn}x_n)
$$

则  

$$\frac{\partial (X^TAX)}{\partial X} = 
\begin{bmatrix}
    \frac{\partial (\mathbf{x}^T\mathbf{A}\mathbf{x})}{\partial x_1}\\
    \frac{\partial (\mathbf{x}^T\mathbf{A}\mathbf{x})}{\partial x_2}\\
    \vdots\\
    \frac{\partial (\mathbf{x}^T\mathbf{A}\mathbf{x})}{\partial x_n}\\
\end{bmatrix}
$$

$$
=
\begin{bmatrix}
    (a_{11}x_1+a_{21}x_2+\cdots+a_{n1}x_n) + (a_{11}x_1+a_{12}x_2+\cdots+a_{1n}x_n)\\
    (a_{12}x_1+a_{22}x_2+\cdots+a_{n2}x_n) + (a_{21}x_1+a_{22}x_2+\cdots+a_{2n}x_n)\\
    \vdots\\
    (a_{1n}x_1+a_{2n}x_2+\cdots+a_{nn}x_n) + (a_{n1}x_1+a_{n2}x_2+\cdots+a_{nn}x_n)\\
\end{bmatrix}
$$

$$
=
\begin{bmatrix}
    (a_{11}x_1+a_{21}x_2+\cdots+a_{n1}x_n)\\
    (a_{12}x_1+a_{22}x_2+\cdots+a_{n2}x_n)\\
    \vdots\\
    (a_{1n}x_1+a_{2n}x_2+\cdots+a_{nn}x_n)\\
\end{bmatrix} 
+
\begin{bmatrix}
    (a_{11}x_1+a_{12}x_2+\cdots+a_{1n}x_n)\\
    (a_{21}x_1+a_{22}x_2+\cdots+a_{2n}x_n)\\
    \vdots\\
    (a_{n1}x_1+a_{n2}x_2+\cdots+a_{nn}x_n)\\
\end{bmatrix} 
$$
$$ = \mathbf{A}^\mathbf{T}\mathbf{x} + \mathbf{A}\mathbf{x} $$

## 根据上述四个矩阵基本求导结果推导最小二乘法
**目标函数** 
$$SSE = J(\mathbf{w}) = \parallel \mathbf{y} - \mathbf{w}\mathbf{x}\parallel_2^2 = (\mathbf{y}-\mathbf{x}\mathbf{w})^T(\mathbf{y} - \mathbf{x}\mathbf{w})$$

$$
=
[\mathbf{y}^T-(\mathbf{x}\mathbf{w})^T](\mathbf{y} - \mathbf{x}\mathbf{w})
$$

$$
=
\mathbf{y}^T\mathbf{y} - \mathbf{y}^T\mathbf{x}\mathbf{w}-(\mathbf{x}\mathbf{w})^T\mathbf{y} +(\mathbf{x}\mathbf{w})^T\mathbf{x}\mathbf{w}
$$

$$
=
\mathbf{y}^T\mathbf{y} - \mathbf{y}^T\mathbf{x}\mathbf{w}-\mathbf{w}^T\mathbf{x}^T\mathbf{y} +\mathbf{w}^T\mathbf{x}\mathbf{x}\mathbf{w}
$$

$$
\frac{\partial J(\mathbf{w})}{\partial \mathbf{{w}}}=
\mathbf{0} - 2\mathbf{x}^T\mathbf{y} - (\mathbf{x}^T\mathbf{x})^T\mathbf{w} + \mathbf{x}^T\mathbf{x}\mathbf{w}
$$
$$
=2(\mathbf{x}^T\mathbf{x}\mathbf{w} + \mathbf{x}^T\mathbf{y})=\mathbf{0}
$$

则

$$
\mathbf{x}^T\mathbf{x}\mathbf{w} = \mathbf{x}^T\mathbf{y}
$$
即
$$
\mathbf{w} = (\mathbf{x}^T\mathbf{x})\mathbf{x}^T\mathbf{y}
$$
$$
s.t. 
\
\ \mathbf{x}^T\mathbf{x} 可逆 (|\mathbf{x}^T\mathbf{x}| \neq 0)
$$

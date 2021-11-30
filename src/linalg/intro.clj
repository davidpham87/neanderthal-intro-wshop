(ns linalg.intro
  (:require
   [nextjournal.clerk :as clerk]
   [uncomplicate.neanderthal.core :as unc]
   [uncomplicate.neanderthal.native :as unn]))

;; # Goals

;;  - Introduce heuristically basic concepts of linear algebra to neophytes with
;;    Clojure knowledge.
;;  - Numerical Linear Algebra is the field concerned with the numerical
;;    application of linear algebra.

;; Caveat: I will consciously ignore a few mathematical details, to emphasize on
;; the intuition [this hurts me as a mathematician].

;; We will cover:

;; - Linear Vector Spaces in Finite Dimension
;; - Linear Mappings
;; - Matrix
;; - Eigenvalues
;; - Special matrices
;; - Neanderthal API.

;; Numerical linear algebra should be a topic by itself, and hence left out for
;; this presentation.

;; # What is Linear Algebra?

;; Intuitively, we want to:

;;   - extend our understanding of geometry in 2D and 3D to higher dimensions
;;   - generalize notions such as distance and isomorphism
;;   - explore if there exist special structures or transformation

;; Crudely speaking and for a software engineer, linear algebra is the science and
;; art of adding and multiplying numbers fast without a user-defined loop.

;; For mathematicians, these are the questions of interest (extend to an arbitrary
;; finite integer number $n$):

;; - Addition:

;; $$ \vec v + \vec w = \begin{bmatrix}
;;       v_{1} & v_{2} & v_{3} \\
;; \end{bmatrix}
;; +
;; \begin{bmatrix}
;;       w_{1} & w_{2} & w_{3}\\
;; \end{bmatrix}
;;     =
;; \begin{bmatrix}
;;       v_{1} + w_{1} & v_{2} + w_{2} & v_{3} + w_{3}
;; \end{bmatrix} $$

;; The computational complexity is $O(n)$.

;; - Scaling:

;; $$ \alpha \vec v = \alpha
;; \begin{bmatrix}
;;       v_{1} & v_{2} & v_{3}
;; \end{bmatrix}
;; =
;; \begin{bmatrix}
;;       \alpha v_{1} & \alpha v_{2} & \alpha v_{3}
;; \end{bmatrix} $$

;; It also has $O(n)$ computational complexity.

;; - Matrix multiplication:

;; $$ AB = \begin{bmatrix}
;;       a_{11} & a_{12} & a_{13} \\
;;       a_{21} & a_{22} & a_{23} \\
;;       a_{31} & a_{32} & a_{33} \\
;;       a_{41} & a_{42} & a_{43}
;; \end{bmatrix}
;; \cdot
;; \begin{bmatrix}
;;       b_{11} & b_{21} \\
;;       b_{21} & b_{22} \\
;;       b_{31} & b_{32}
;; \end{bmatrix}
;;     =
;; \begin{bmatrix}
;;       r_{11} & r_{12} \\
;;       r_{21} & r_{22} \\
;;       r_{31} & r_{32} \\
;;       r_{41} & r_{42}
;; \end{bmatrix} = R$$

;; $$ r_{ij} = \sum_{k} a_{ik}b_{kj} $$

;; The computational complexity of $O(nmp)$ speed, $n$ count of rows for the first
;; matrix, $m$ count of the rows for the second matrix, $p$ number of columns
;; of the second matrix.

;; When the second matrix is a single column, we have $O(n^2)$ complexity algorithm,
;; and when both matrices have the number of rows and columns $O(n^3)$.

;; In code, we want to compute the following operations as fast as possible:


(def v [0 1 2 3])
(def w [3 2 -1 0])
(def a 3)
(mapv + v w) ;; [3 3 1 3]
(mapv (partial * a)  v) ;; [0 3 6 9]
(def X [[1 1 1 1]
        [2 2 2 2]])
(mapv #(reduce + (map * % v)) X) ;; [6 12]
(mapv #(mapv (fn [x] (reduce + (map * % x))) [v w]) X)
;; [[6 4]
;;  [12 8]]


;; Mathematically speaking, linear algebra studies vector spaces and linear
;; transformations between vector spaces (sets of things with some structures).

;; The question that we can ask ourselves:

;; - How can we interpret the vector and the matrices? (vector spaces, and linear
;;   transformation).
;; - In a matrix, are all the rows/columns containing additional information? (rank).
;; - Is there a more efficient representation of the matrices?

;; # Why Should We Care about Linear Algebra?

;; These are examples of areas where linear algebra is applied:

;; - Statistics: linear algebra provides a vocabulary and
;;   tools to solve problems in high dimensions.

;; - Machine Learning: linear algebra is abused for computer vision, natural
;;   language processing, speech recognition, and gaming agent.

;; - Simulations of [stochastic] systems (Markov chains): when studying systems
;;   where variables can influence each other, linear algebra might help to find
;;   the equilibrium state.

;; - Study of graphs/networks: graphs and networks can be represented as a matrix
;;   where each row represents the outward connection from a vertex to another
;;   vertex.

;; - Other examples: solving linear equations, solving partial differential
;;   equations, optimization and budgeting, computer graphics.

;; # Why Learn a Specialized Linear Algebra Library?

;; One could heuristically define most linear algebra operations as such


(def plus #(apply mapv + %&))
(defn times [alpha x] (mapv (partial * alpha) x))
(defn dot [x y] (reduce + (mapv * x y)))
;; Suppose matrix are row majors
(defn transpose [Y] (apply mapv vector Y))
(defn %*% [X Y]
  (mapv (fn [x] (mapv #(dot x %) (transpose Y))) X))

(plus [1 2] [0 1])
(times 3 [1 2])
(dot [1 2] [0 1])

(%*% [[1 0] [0 1]]
     [[1 2] [3 4]])

(%*% [[0 1] [1 0]]
     [[1 2] [3 4]])

(%*% [[-1 0] [0 1]]
     [[1 2] [3 4]])


;; ## Speed and Algorithms

;; The reason linear algebra is a necessity in practice for these fields is thanks
;; to the concept and the hardware implementation of *single instruction multiple
;; data* (`SIMD`), in other words, the art of applying the same operation to
;; multiple data points. CPU and GPU have specialized instructions to perform
;; linear algebra operations, speeding them up by a magnitude of orders
;; (depending on the size of the problem).

;; The gain speed allowed for the creation of algorithms (such as the bootstrap in
;; statistics).

;; Depending on the structure of your problem, you could leverage the shape of
;; your matrices to speed up, even more, the computations.

;; # Linear Algebra Concepts

;; ## Vector Spaces Example:

;; Take two vectors of numbers and a scalar say

(def v [0 1 2 3])
(def w [3 2 -1 0])
(def a 3)

;; Then, intuitively the addition and the scaling operation are
;; defined as element-wise operations:

(mapv + v w)
(mapv (partial * a)  v)

;; We can try to abstract this behavior, this would lead to the proper definition
;; of a vector space in mathematics.

;; ## Vector Spaces: Definition

;; A vector space, a set of elements, is a set stable under a `+` operation and a
;; `*` scalar operation, with the following property, for all $u, v, w \in V$:

;; - Commutativity: $u+v = v+u$
;; - Associativity: $(u + v) + w = u (v + w)$ and $(ab)v = a(bv)$.
;; - Additive identity: there exists $0 \in V$ such that $v+0 = v$.
;; - Additive inverse: for all $v \in V$, there exist $w \in W$ such that $v+w =
;;   0$.
;; - Multiplicative identity ($1v=v$, for all $v \in V$),
;; - Distributive properties: $(\alpha+\beta)(u+v) = \alpha u + \alpha u + \beta v + \beta v$, for all $\alpha, \beta \in \mathcal{F}$, and $v, w \in V$.

;; The most used vector space is probably the cartesian product of $\mathbb{R}$,
;; that is $\mathbb{R} \times \mathbb{R} = \mathbb{R}^2$, and generally
;; $\mathbb{R}^n$.

;; Other vector spaces: the space of function from $\mathbb{R}$ to
;; $\mathbb{R}$.

;; ### Distance and Angle: Inner Product Space & Norm

;; An inner product on $V$ is a function taking each ordered pair $(u, v)$ of
;; elements in $V$ to a number $\langle u, v \rangle$ \in $\mathbb{F}$ with the
;; following properties:

;; - Positivity $\langle v, v \rangle \geq 0$ for all $v \in V$.
;; - Definiteness $\langle v, v \rangle = 0$ if and only if $v = 0$;
;; - Additivity and homogeneity in the first slot: for all $u, v, w \in V$ and
;;   $\lambda \in \mathbb{F}$

;;   $$\langle \lambda u + v, w \rangle = \lambda \langle u, w \rangle + \langle v, w \rangle$$

;; - Conjugate symmetry: $\langle u, v \rangle$ is the conjugate of $\langle v, u
;;   \rangle$ (when $\mathbb{F} = \mathbb{R}, they are equal).

;; In most applications, the inner product of two vectors is the euclidean dot
;; product $\langle v, w \rangle = \sum_i v_i w_i$, which translate to


(defn dot [v w] (reduce + (map * v w)))


;; Usually, one uses the notation $v^\top w$ for the euclidean product instead of
;; $\langle v, w \rangle$.

;; Thanks to the inner product, we can generalize the notion of *length* $\| v \|$ (we call
;; it a *norm*) for a vector $v$ as

;; $$\| v \| = \sqrt{\langle v, v \rangle}.$$

;; This norm then define the *distance* between two vectors $v$ and $w$ as the
;; norm of their difference:

;; $$\| v - w \| = \sqrt{\langle v-w, v-w \rangle}.$$

;; ### Subspaces

;; Let's take the space $V=\mathbb{R}^3$ as an example.

;; We can define regions (in $\mathbb{R}^3$ they will be lines or plane surface) in the
;; space that are strictly smaller than $V$ but would remain subvector
;; space. For example,

;; $$\{ \lambda (1, 0): \lambda \in \mathbb{R}\}$$

;; $$\{ \lambda (1, 0) + \mu (0, 1): \lambda, \mu \in \mathbb{R}\}$$

;; Formally a subvector space is a subset of $V$ (possibly equal to $V$) which is
;; also a vector space. Common examples of subvector space are projections.

;; ## Linear Independence, Basis and Dimension

;; We say a list $v_1, \dots, v_n$ is *linearly independent* if $\sum_{i=1}^n \alpha_i v_i = 0$,
;; implies all $\alpha_i$ are equal to 0. Basically, it means all the
;; vectors contains some information.

;; We say a vector space $V$ is in the span $v_1, \dots, v_n$ if any element in
;; $V$ can be represented as a linear representation of $v_1, \dots, v_n$. That is
;; for all $v \in V$, there exist a list of $\alpha_1, \dots, \alpha_n$ such that

;; $$v = \sum_{i=1}^n \alpha_i v_i$$

;; We say the list is a basis of $V$ if it is linearly independent.

;; Finally, one defines the dimension of $V$ as the number of elements in any basis
;; of $V$.

;; ### Orthonormal basis

;; A list $(e_1, \dots, e_n)$ of vectors in $V$ is *orthonormal* if

;; $$ \langle e_j, e_k \rangle = \mathbb{1}(j=k) \textrm{ for all } j,k=1, \dots, n. $$

;; Intuitively, the vector $e_j$ are *perpendicular* to each other and have a
;; *norm* or *length* of 1. There might exist an infinite amount of orthonormal
;; bases for a given space, but their count is always the same.

;; The most widely used orthonormal basis is the canonical basis $e_1, \dots, e_n$,
;; where each vector $e_j$ contains only 0, except at position
;; $j$, where the number is $1$. For example in $\mathbb{R}^3$,

;; $$ e_1  = \begin{bmatrix} 1 & 0 & 0 \end{bmatrix}, \quad e_2 = \begin{bmatrix} 0 & 1 & 0\end{bmatrix}, \quad e_3 = \begin{bmatrix} 0 & 0 & 1\end{bmatrix} $$

;; # Linear Mappings and Matrices

;; A *linear map* from a space $V$ to another vector space $W$ is a function $T: V
;; \to W,$ such that for all $u, v \in V, \lambda \in \mathcal{F}$ one has

;; $$ T(\lambda v + w) = \lambda T(v) + T(w). $$

;; One such example in $\mathbb{R}^2$: the mapping the switch the sign of the
;; second component:

;; $$ T(\begin{bmatrix} v_1 & v_2 \end{bmatrix}) = \begin{bmatrix} v_1 & -v_2 \end{bmatrix} $$

;; Another example of mapping: the derivative of a function in the space of
;; functions that can be derived is a linear operation, as

;; $$T(\alpha f+ \beta g) = (\alpha f + \beta g)' = \alpha f' + \beta g' = \alpha
;; T(f) + \beta T(g).$$

;; The space of linear maps from $V \to W$ is also a vector space! In other words, if $S$
;; and $T$ are linear map from $V \to W$, $\lambda \in \mathbb{F}$ then $\lambda S+T$
;; defined as

;; $$ (\lambda S+T)(u) = \lambda S(u) + T(u) $$

;; is also a linear map from $V \to W$.


;; ## Composition of linear maps

;; We can chain linear map, if the space are coinciding: if $T: U \to V$, $S: V \to W$,
;; then the product $ST: U \to W$ defined as $(ST)(u) = S(T(u))$ is a
;; linear map from $U \to W$.

;; ## Matrices

;; But why is all of this useful for understanding matrices?!

;; - The space of linear maps and the space matrices are bijective, that is, we can
;;   represent every linear map as a matrice and inversely.
;; - The space of matrices are vector spaces.

;; A $m \times n$ matrix with $m$ rows and $n$ columns is a rectangular array of number:

;; $$ A = \begin{bmatrix} a_{11} & \cdots & a_{1n} \\  \vdots & & \vdots \\ a_{m1} & \cdots & a_{mn}\end{bmatrix}$$


;; ## Matrice of a linear map

;; Let $T: V \to W$, $(v_1, \dots, v_n)$, resp. $(w_1, \dots, w_m)$, be a basis
;; for $V$, resp. $W$. Then, we can compute for every $v_j$

;; $$ T(v_j) = a_{1j} w_1 + \dots + a_{mj} w_m $$

;; The $m \times n$ matrix $\mathcal{M}(T)$ of a linear transformation $T$ is then
;; defined as

;; $$ \mathcal{M}(T) = \begin{bmatrix} a_{11} & \cdots & a_{1n} \\ \vdots & &
;; \vdots \\ a_{m1} & \cdots & a_{mn}\end{bmatrix} $$

;; Noteworthy:

;; - It is easier to interpret the matrix columnwise: the coefficient from a
;;   column $j$ is the coefficient of the linear combination to express $w=T(v_j)$
;;   in the basis of $w_1, \dots, w_m$.
;; - The relation is bijective: for every matrix $A$, there exist a linear mapping
;;   $T_A$ from $\mathbb{R}^n$ to $\mathbb{R}^m$, such that $T_A(v_j) = \sum_i
;;   \alpha_{ij} w_i$.
;; - With this connection, the space of matrices is also a vector space! One
;;   defines the addition of matrices as element-wise addition. That is for two
;;   matrices $A,B$ and a coefficient $\lambda$

;;   $$ (\lambda A + B)_{ij} = \lambda a_{ij} + b_{ij},$$

;;   since we could associate $\lambda A + B$ to the mapping $T_{\lambda A + B}.$

;; ## Matrix of a composition and the Product of Matrices

;;   Remember we defined the composition $(S_AT_B): U \to W$ of
;;   $S_A: U \to V$ and $T_B: V \to W$ as $(S_AT_B)(u) = S_A(T_B(u))$. Let $(u_1, \dots, u_n)$
;;   , $(v_1 \dots, v_p)$, $(w_1, \dots, w_m)$ be basis of $U$, $V$,
;;   resp. $W$, and $A=\mathcal{M}(S_A)$, $B=\mathcal{M}(T_B)$. Then for every $u_j$

;;   $$
;;   \begin{aligned} S(T(u_j)) & = S\left(\sum_{l=1}^p b_{lj} v_l\right) \\
;;    & = \sum_{l=1}^p b_{lj} S(v_l) \\
;;    & = \sum_{l=1}^p b_{lj} \sum_{i=1}^m a_{il} w_i\\
;;    & = \sum_{i=1}^m \left(\sum_{l=1}^p  a_{il} b_{lj}\right) w_i \\
;;    & = \left(\sum_{l=1}^p  a_{1l} b_{lj}\right) w_1 + \dots + \left(\sum_{l=1}^p  a_{ml} b_{lj}\right) w_m
;;   \end{aligned}
;;   $$

;;   As such, the element $(i,j)$ of the matrix of the $S_AT_B$ is given by

;;   $$ (ST)_{ij} =  \sum_{l=1}^p a_{il} b_{lj}.$$

;;   Since we have a one-to-one relationship with the space of matrices, we could
;;   define a new operation: the matrix multiplication. Hence for a matrix
;;   $A$ and $B$, such that the count of columns of $A$ matches the count of rows
;;   in $B$, we define the matrix multiplication as $C = AB$ such that

;;   $$ c_{ij} = \sum_{l=1}^p a_{il} b_{lj}.$$

;;   Note that the dimensions must match, and that the operation is not
;;   commutative in general ($AB \neq BA$).

;; ## Example

;; Rotation in the plan is also a linear map (a fact that is quite used in
;; computer vision). Its corresponding matrix in a canonical basis is given by

;; $$ \begin{bmatrix}
;; \cos(\theta) & -\sin(\theta)\\
;; \sin(\theta) & \cos(\theta)\\
;; \end{bmatrix}
;; $$

;; where $\theta$ describes the rotation of the points. Interestingly, shifting
;; points in $\mathbb{R}^2$ can't be described in $\mathbb{R}^2$ but could in
;; $\mathbb{R}^3$ with the following matrices:

;; ![Affine Transformation as a Linear Map](https://upload.wikimedia.org/wikipedia/commons/thumb/2/2c/2D_affine_transformation_matrix.svg/1920px-2D_affine_transformation_matrix.svg.png)


;; ## Special matrices structures

;; - Diagonal matrices: all the elements of the matrices are 0, except those on the diagonal. Example:

;; $$ \begin{pmatrix}
;;     3 & 0 \\
;;     0 & 2
;;     \end{pmatrix}$$

;; - Block/Band diagonal matrices:

;; $$ \begin{pmatrix}
;;     A & & *\\
;;     &B\\
;;     * &&C\\ \end{pmatrix}$$

;; where $A, B, C$ are square matrices. $*$ inside matrices are the notation for
;; saying the elements are equal to 0.

;; - Upper and Lower Triangular Matrices: all elements below or above the diagonal
;; are equal to 0. Example:

;; $$ \begin{pmatrix} 3 & 3 & 1 \\  & 2 & 0 \\ * &  & 2 \end{pmatrix} $$

;; - Symmetric matrices: $A=A^\top$ (the transpose $A^\top$ of the matrice $A$ is
;;   defined as $a_{ij} = a_{ji}^\top$).

;; Each of these structures allows for calculation optimization. Thus our
;; challenges is often to express the problem at hand in these forms.

;; <!-- ## Least Square -->

;; <!-- Small trick: Neanderthal API and Machine Learning/Statistic notations do not -->
;; <!-- coincide. In the latter, the design matrix and regression parameters are -->
;; <!-- denoted with $X$ and $\beta$. In contrast Neanderthal uses $A$ and $x$. -->

;; # Eigenvalues and Eigenvectors

;; Let $T: V \to V$ be a linear map. In order to save computation, can we ask if
;; there exists strict subspaces of $V$ where $T$ would be invariant, that is
;; $T(v) = \lambda v$, for some $\lambda \neq 0$ and $v \neq \vec 0$. If such
;; $\lambda$ exists, we call it a eigenvalue, and the associated $v$ a
;; eigenvectors. Note, we usually take the convention that $\| v \| = 1$.

;; The eigenvectors are important as they describe an invariant space, such as if
;; $v_1, v_2$ are eigenvectors with respective eigenvalues $\lambda_1, \lambda_2$,
;; then if $v \in \textrm{span}(v_1, v_2)$, then

;; $$ \begin{aligned}
;;   T(v) & = T(\alpha_1 v_1 + \alpha_2 v2) = \alpha_1 T(v_1) + \alpha_2 T(v_2) \\
;;        & = \alpha_1 \lambda_1 v_1 + \alpha_2 \lambda_2 v_2 \in \textrm{span}(v_1, v_2)
;;  \end{aligned}
;; $$

;; Eigenvectors allow for understanding the linear map more easily. They are the
;; directions along which the linear map is just *dilatation/compression*, and/or
;; *flipping*, eigenvalues are the factors of compression. The more directions,
;; the better the understanding. It allows for decoupling the behavior of each
;; rows with each other.

;; # Neanderthal API

;; One can apply all the previous topics to any linear algebra library, we are
;; now going to dig into the main concepts of Neanderthal's API.  In essence, it
;; follows [LAPACK](https://en.wikipedia.org/wiki/LAPACK) naming convention.

;; ## Construction of Matrices Vector

;; There are three dimensions for understanding the API:

;; - Numbers type: (`d`ouble, `f`loat (single precision), `i`nteger)
;; - Matrices structure: `v` (vector), `ge` (general/dense matrices), `sy` (symmetric),
;;   `tr` (triangular).
;; - Runtime (Native `uncomplicate.neanderthal.native`, CUDA `uncomplicate.neanderthal.cuda`,
;;   OpenCL `uncomplicate.neanderthal.opencl`).

;; Hence to create a double general/dense matrix using native (Intel MKL), we
;; would need to call `uncomplicate.neanderthal.native/dge`, similarly for a
;; double vector `uncomplicate.neanderthal.native/dv`.

;; ## Functions on Matrices

;; - Basic operations (altering variables, and many basic linear algebra
;;   operations) are exposed in `uncomplicate.neanderthal.core`.
;; - Advanced math operations on matrices and vectors are exposed in
;;   `uncomplicate.neanderthal.vect-math`.
;; - Linear Algebra Algorithms (`uncomplicate.neanderthal.linalg`): finding eigenvalue,
;;   solving, least squares, matrices decomposition.
;; - Pure against Destructive API: functions finishing with a bang `!` will mutate
;;   their argument and save time and possibly speed up the computation.

;; # Main Takeaways

;; - Linear algebra is the study of linear relationships (addition and scaling) in
;;   arbitrary finite dimensions.
;; - Adding and subtracting matrices and vectors are $O(n)$ operations.
;; - Multiplying a vector with a matrix is a $O(n^2)$.
;; - Multiplying matrices (composing linear maps) is $O(n^3)$.

;; - Vector spaces are mainly generalization of our intuition of geometry in 2D
;;   and 3D, where we only consider addition and scaling.
;; - Vector spaces are spanned by a basis, a list of linearly independent vectors.
;; - Linear maps are transformation that preserves the additive and homogeneity
;;   between the domain and image space.
;; - Linear maps can be described by the coefficient of the image of the basis of
;;   the input space.

;; - There exists a one-to-one relationship between the space of linear maps and the
;;   space matrices. Hence matrices represent linear transformation.
;; - For a given linear map, there might exist a basis more appropriate
;;   to perform the transformation.

;; - There are special matrices, which could accelerate your computation.
;; - Eigenvectors of a linear map describe invariant spaces of the mapping.

;; - To get the best performance of Neandterhal, you should keep the data
;;   structure in Neandterhal's vectors/matrices. Avoid converting data between
;;   Clojure and Neandterhal memory layout.

---
title: Introduction to Linear Algebra
...

# Strategy

 Going from example first, then abstract.

# Goals

 - Introduce heuristically basic concepts of linear algebra to neophytes with
   Clojure knowledge.
 - Numerical Linear Algebra is the field concerned with the numerical
   application of linear algebra.

Caveat: I will consciously ignore a few mathematical details, to emphasize on
the intuition [this hurts me as a mathematician].

We will cover:

- Linear Vector Spaces in Finite Dimension
- Linear Mappings
- Matrix
- Eigenvalues
- Special matrices
- Neanderthal API.

Numerical linear algebra should be a topic by itself, and hence left out for
this presentation.

# What is linear algebra?

Crudely speaking and for a software engineer, linear algebra is the science and
art of adding and multiplying numbers fast without a user-defined loop.

Mathematically, linear algebra studies vector spaces and linear transformation
between spaces.

# Why caring about linear algebra?

These are examples of area where linear algebra is applied the most are:

- Statistics: linear algebra provides a vocabulary and
  tools to solve problem in high dimensions.

- Machine Learning: linear algebra is abused for computer vision, natural
  language processing, speech recognition, and gaming agent.

- Simulations of [stochastic] systems (Markov chains): when studying systems
  where variable can influence each other, linear algebra might help to find
  the equilibrium state.

- Study of graphs/networks: graphs and networks can be represented as a
  matrix where each row represent the outward connection from a vertex to
  another vertex.

- Other ares: solving partial differential equations, optimization and
  budgeting, computer graphics [more details later].

# Why learning a specialized Linear Algebra Library?

It is true that one could heuristically define most linear algebra operations
as such

``` clojure
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
```

## Speed and Algorithms

The reason linear algebra is a necessity in practice for these fields is thanks
to the concept and the hardware implementation of *single instruction multiple
data* (`SIMD`), in other words the art of applying the same operation to
multiple data points. CPU and GPU have specialized instructions to perform
linear algebra operations which speeds up operations by a magnitude of orders
(depending on the size of the problem).

The gain speed allowed to create algorithms (such as the bootstrap in
statistics).

Depending on the structure of your problem, you could leverage the shape of
your matrices to speed up, even more, the computations.

# Linear Algebra Concepts

## Vector Spaces Example:

Take two vectors of numbers and a scalar say

```clojure
(def v [0 1 2 3])
(def w [3 2 -1 0])
(def a 3)
 ```

Then intuitively the addition and the scaling operation are
defined as:

``` clojure
(map + v w) ;; [3 3 1 3]
(map (partial * a)  v) ;; [0 3 6 9]
```

We can try to abstract this behavior, this would lead to the proper definition
of a vector space in mathematics.

## Vector Spaces

A vector space, a set of elements, is a set stable under a `+` operation and a
`*` scalar operation, with the following property, for all $u, v, w \in V$:

- Commutativity: $u+v = v+u$
- Assioativity: $(u + v) + w = u (v + w)$ and $(ab)v = a(bv)$.
- Additive identity: there exists $0 \in V$ such that $v+0 = v$.
- Additive inverse: for all $v \in V$, there exist $w \in W$ such that $v+w =
  0$.
- Multiplicative identity ($1v=v$, for all $v \in V$),
- Distributive properties: $(\alpha+\beta)(u+v) = \alpha u + \alpha u + \beta
  v + \beta v$, for all $\alpha, \beta \in \mathcal{F}$, and $v, w \in V$.

The most used vector space is probably the cartesian product of $\mathbb{R}$,
that is $\mathbb{R} \times \mathbb{R} = \mathbb{R}^2$, and generally
$\mathbb{R}^n$.

Other vector spaces: the space of function from $\mathbb{R}$ to
$\mathbb{R}$.

### Inner Product Space

An inner product on $V$ is a function taking each ordered pair $(u, v)$ of
elements in $V$ to a number $\langle u, v \rangle$ \in $\mathbb{F}$ with the
following properties:

- Positivity $\langle v, v \rangle \geq 0$ for all $v \in V$.
- Definiteness $\langle v, v \rangle = 0$ if and only if $v = 0$;
- Additivity and homogeneity in the first slot: for all $u, v, w \in V$ and
  $\lambda \in \mathbb{F}$

  $\langle \lambda u + v, w \rangle = \lambda \langle u, w \rangle + \langle v, w \rangle$

- Conjugate symmetry: $\langle u, v \rangle$ is the conjugate of $\langle v, u
  \rangle$ (when $\mathbb{F} = \mathbb{R}, they are equal).

In most applications, the inner product of two vectors is

```clojure
(reduce + (map * %1 %2))
```

### Subspace and basis

Let's take the space $V=\mathbb{R}^3 as an example.

# TODO Make a an empty plot

We can define regions (in this they will be lines or plane surface) in the
space that are strictly smaller than $V$ but would still remain subvector
space. Example are

$$\{ \lambda (1, 0): \lambda \in \mathbb{R}\}$$

$$\{ \lambda (1, 0) + \mu (0, 1): \lambda, \mu \in \mathbb{R}\}$$

Formally, a subvector space is a subset of $V$ (possibly equal to $V$) which is
also a vector space. Common example of subvector space are projections.

## Linear Independency, Basis and Dimension

We say a list $v_1, \dots, v_n$ is *linearly independent* if $\sum_i \alpha_i
v_i = 0$, implies all $\alpha_i$ are equal to 0. Basically, it means all the
vectors contains some information.

We say a vector space $V$ is in the span $v_1, \dots, v_n$ if any element in
$V$ can be represented as a linear representation of $v_1, \dots, v_n$. That is
for all $v \in V$

$v = \sum_i \alpha_i v_i$

for some $\alpha = (\alpha_1, \dots, \alpha_n$. We say the list is basis of $V$
if it is linearly independent.

Finally, the dimension of $V$ is defined as the number of element in any basis
of $V$.

### Orthonormal basis

A list $e_1, \dots, e_n$ of vectors in $V$ is *orthonormal* if

$\langle e_j, e_k \rangle = \mathbb{1}(j=k)$ for all $j,k=1, \dots, n$.

## Linear Mappings

Another example of mapping: the derivative of a function in the space of
function that can be derived is a linear operation, as

$T(\alpha f+ \beta g) = (\alpha f + \beta g)' = \alpha f' + \beta  g' = \alpha T(f) + \beta T(g)$.

Let $T\in \mathcal{L}(V)$, with $V$ inner product space. Then $T^*$ is an
*adjoint* if $< Tv, w> = <v, T^*w>$.

An operator is *self-adjoint* if $T=T*$.

Normal operator on a inner producte space is called *normal* if it commutes
with its adjoint. If $T\in \mathcal{L}(V)$ is normal if $TT^* = T^*T$.

# Matrices

- But why is all of this useful for understanding matrices?!
- The space of linear maps and the space matrices are bijective, that is we can
  represent every linear map as a matrice and inversely.
- The space of matrices are vector spaces.

## Least Square

Small trick: Neanderthal API and Machine Learning/Statistic notations do not
coincide. In the latter, the design matrix and regression parameters are
denoted with $X$ and $\beta$. In contrast Neanderthal uses $A$ and $x$.

# Main Takeaways

- Linear algebra is the study of linear relationships (addition and scaling).
- The space of linear maps and the space matrices are bijective.

- Adding and substracting matrices and vector are $O(n)$ operations.
- Multiplying a vector with a matrix is a $O(n^2)$.
- Multiplying matrices (composing linear maps) is $O(n^3)$.

- There are special matrices which could accelerate your computation.
- To get the best performance of Neandterhal, you should keep the data
  structure in Neandterhal's vectors/matrices. Avoid converting data between
  clojure and Neandterhal memory layout.


# Plan

- What is Linear Algebra? Start with example of geometry [vector/planes], then
  speak about linear transformation as mappings/functions [speak about
  projection, bijectivity], Transition to matrix showing a one to one
  relationship between function to matrices. Make the case of composition of
  mapping is multiplication of matrices?

- Start making example with scaling and rotation, extend to translate.

- Linear Algebra in Clojure: speak about Neanderthal API and link to numpy?

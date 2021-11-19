---
title: Introduction to Linear Algebra
...

# Introduction to Linear Algebra

# Strategy

 Going from example first, then abstract.

# Goals

 - Introduce heuristically basic concepts of linear algebra to neophytes with
   Clojure knowledge.

Caveat: I will consciously ignore a few mathematical details, to emphasize on
the intuition [this hurts me as a mathematician].

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

## Vector Spaces

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

- Conjugate symmetry: $\langle u, v \rangle = \bar \langle v, u \rangle$

In most applications, the inner product of two vectors is

```clojure
(reduce + (map * %1 %2))
```

### Subspace and basis

### Orthonormal basis

A list $e_1, \dots, e_n$ of vectors in $V$ is *orthonormal* if

$\langle e_j, e_k \rangle = 1(j=k)$




## Linear Mappings

Another example of mapping: the derivative of a function in the space of
function that can be derived is a linear operation, as

$T(\alpha f+ \beta g) = (\alpha f + \beta g)' = \alpha f' + \beta  g' = \alpha T(f) + \beta T(g)$.

Let $T\in \mathcal{L}(V)$, with $V$ inner product space. Then $T^*$ is an
*adjoint* if $< Tv, w> = <v, T^*w>$.

An operator is *self-adjoint* if $T=T*$.

Normal operator on a inner producte space is called *normal* if it commutes
with its adjoint. If $T\in \mathcal{L}(V)$ is normal if $TT^* = T^*T$.

## Least Square

Small trick: Neanderthal API and Machine Learning/Statistic notations do not
coincide. In the latter, the design matrix and regression parameters are
denoted with $X$ and $\beta$. In contrast Neanderthal uses $A$ and $x$.


# Plan

- What is Linear Algebra? Start with example of geometry [vector/planes], then
  speak about linear transformation as mappings/functions [speak about
  projection, bijectivity], Transition to matrix showing a one to one
  relationship between function to matrices. Make the case of composition of
  mapping is multiplication of matrices?

- Start making example with scaling and rotation, extend to translate.

- Linear Algebra in Clojure: speak about Neanderthal API and link to numpy?

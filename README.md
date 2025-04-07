# ENM5320
AI4Science/Science4AI Spring 2025

MW 10:15am-11:44am in TOWN 309 (1/15 to 4/30)

Trask OH: 5th floor AGH Tues. 915-1030

Grad TA OH: AGH 214, Friday, 4pm-6pm.

Ed Discussion board: https://edstem.org/us/courses/74501

# Current Schedule
- **Jan 15.**  Quick intro to pytorch and course logistics
- **Jan 20.** No class for holiday
- **Jan 22.** Finite difference crash course. Fourier analysis, grid functions, analysis for solutions to linear transport.
  - [Lecture notes](https://github.com/natrask/ENM5320/blob/main/Lecture%20Notes/Lecture_2.pdf)
  - [Homework assignment](https://github.com/natrask/ENM5320/blob/main/Assignments/ENM5320__HW1.pdf). Due 1/29 by midnight through Canvas.
  - [Jupyter notebook](https://github.com/natrask/ENM5320/blob/main/Code/Lecture01.ipynb)
  - [E-book reference](https://find.library.upenn.edu/catalog/9977071082103681?hld_id=53499053800003681). Material is drawn from Ch. 1-2 of Gustafsson.
- **Jan 29.** Virtual lecture. Introduction to data-driven models. Finite difference code and pytorch review.
  - [Link to lecture video](https://www.youtube.com/watch?v=z7enURoFU3k)
- **Jan 31.** Stability analysis and the Lax equivalence theorem (aka fundamental theorem of finite differences)
  - [Lecture notes](https://github.com/natrask/ENM5320/blob/main/Lecture%20Notes/Lecture_4.pdf)
  - [Example code for finite differences in numpy](https://github.com/natrask/ENM5320/blob/main/Code/finiteDifferenceExample.ipynb)
- **Feb 3.** Designing learning architectures with consistency and stability guarantees 
  - [Lecture notes](https://github.com/natrask/ENM5320/blob/main/Lecture%20Notes/Lecture_5.pdf)
  - [Example code for learning stencils in pytorch](https://github.com/natrask/ENM5320/blob/main/Code/PyTorchFDM.ipynb)
  - [Homework assignment](https://github.com/natrask/ENM5320/blob/main/Assignments/ENM5320__HW2.pdf). Due 2/10 by midnight through Canvas.
- **Feb 5.** Nonlinear stability analysis, constrained quadratic programming, and polynomial reproduction.
  - [Lecture notes](https://github.com/natrask/ENM5320/blob/main/Lecture%20Notes/Lecture_6.pdf)
  - [Lecture notes - addendum](https://github.com/natrask/ENM5320/blob/main/Lecture%20Notes/Lecture_6_addendum.pdf)
- **Feb 10.** Coding tutorial coding up nonlinear stencils.
  - [Code](https://github.com/natrask/ENM5320/blob/main/Code/Trainable_Nonlinear_Stencil.ipynb)
  - [Youtube lecture](https://www.youtube.com/watch?v=U6bb5Fv-i-c)
- **Feb 12.** Hamiltonian systems and discrete gradients.
  - [Lecture notes](https://github.com/natrask/ENM5320/blob/main/Lecture%20Notes/Lecture_7.pdf)
- **Feb 17.** Lagrangian mechanics, functional derivatives, the principle of least action, Legendre transforms and Noether's theorem.
  - [Lecture notes](https://github.com/natrask/ENM5320/blob/main/Lecture%20Notes/Lecture_8.pdf)
  - [Feynmann lecture on least action](https://www.feynmanlectures.caltech.edu/II_19.html)
  - [Lagrangian neural networks](https://greydanus.github.io/2020/03/10/lagrangian-nns/)
- **Feb 19.** Obtaining discrete structure-preserving stencils from the principle of least action
  - [Lecture notes](https://github.com/natrask/ENM5320/blob/main/Lecture%20Notes/Lecture_9.pdf)
  - [Additional short proof for Noethers theorem](https://github.com/natrask/ENM5320/blob/main/Lecture%20Notes/noether_thm_simple_proof.pdf)
  - [Homework assignment](https://github.com/natrask/ENM5320/blob/main/Assignments/ENM5320__HW3.pdf). Due 2/26 by midnight through Canvas.
- **Feb 24.** Our complete data-driven hyperbolic system model. Time integration revisited: multi-stage/multi-step schemes, linear stability analysis, Stormer-Verlet
  - [Lecture notes](https://github.com/natrask/ENM5320/blob/main/Lecture%20Notes/Lecture_10.pdf)
  - [Code](https://github.com/natrask/ENM5320/tree/main/Code/HNN_demo)
- **Feb 26.** Finishing last lecture, beginning midterm group miniprojects.
- **Mar 3.** Brief probability review. Maximum likelihood. Stochastic processes, Euler-Maruyama, and structure preserving stochastic dynamics.
  - [Lecture notes](https://github.com/natrask/ENM5320/blob/main/Lecture%20Notes/Lecture_11.pdf)
  - [Reference class notes](/https://chrisrackauckas.com/assets/Papers/ChrisRackauckas-IntuitiveSDEs.pdf)
  - Specific textbooks to fill in probability background are given in 3/3 lecture notes.
- **Mar 17.** Concluding finite difference method. Introduction to Galerkin/Rayleigh-Ritz method and FEM code tutorial.
  - [Lecture notes](https://github.com/natrask/ENM5320/blob/main/Lecture%20Notes/Lecture_12.pdf)
  - [Code](https://github.com/natrask/ENM5320/blob/main/Code/finiteElement.py)
- **Mar 19.** Quasi-optimality estimates, nodal FEM and the interpolant, optimal convergence in L2/H1, abstract Lax-Milgram theory.
  - [Lecture notes](https://github.com/natrask/ENM5320/blob/main/Lecture%20Notes/Lecture_13.pdf)
  - [Code](https://github.com/natrask/ENM5320/blob/main/Code/FEMconvergence.ipynb)
  - References: [Johnson](https://www.amazon.com/Numerical-Solution-Differential-Equations-Mathematics/dp/048646900X), [Brenner & Scott](https://link.springer.com/book/10.1007/978-0-387-75934-0)
- **Mar 24/26.** Application of Lax-Milgram. Reaction-diffusion, elasticity, and incompressibility. Mixed finite element methods.
  - [Lecture notes](https://github.com/natrask/ENM5320/blob/main/Lecture%20Notes/Lecture_14.pdf)
- **Mar 31/Apr 1.** Mixed finite element methods continued. Inf-sup compatibility. Conservation structure and nonlinear perturbations.
  - [Lecture notes](https://github.com/natrask/ENM5320/blob/main/Lecture%20Notes/Lecture_15.pdf)
  - [Slides](https://github.com/natrask/ENM5320/blob/main/Lecture%20videos/slides_4_3.pptx)
- **Apr 7.** Introduction to graphs, analysis of graph Laplacian, graph exterior calculus.
  - [Lecture notes](https://github.com/natrask/ENM5320/blob/main/Lecture%20Notes/Lecture_16.pdf)
  - References: [Spectral graph theory, Chung](https://books.google.com/books?hl=en&lr=&id=4IK8DgAAQBAJ&oi=fnd&pg=PP1&dq=spectral+graph+theory+chung&ots=Et6QXiwStk&sig=eW5_hSCo5ZViVTvpQgKz43nJ00c#v=onepage&q=spectral%20graph%20theory%20chung&f=false), [Statistical ranking and combinatorial Hodge theory, Jiang et al](https://link.springer.com/article/10.1007/s10107-010-0419-x), [Exact physics on graphs, Trask et al](https://www.sciencedirect.com/science/article/pii/S0021999122000316)
    
# Description
Many seek to replicate the successes of AI/ML in computer vision and natural language processing in the sciences, aiming to
tackle previously inaccessible problems in scientific discovery, engineering prediction, and optimal design. ML however has been
powered by “black-box” architectures specifically tailored toward text/image data which lack the mathematical structure
necessary to provide predictions which meet the requirements for high-consequence science and engineering: consistency with
physical principles, numerical robustness, interpretability, and amenability to uncertainty quantification. In this course we will
survey theories of variational mechanics, geometric dynamics and numerical analysis to understand how to construct simulators
from data which respect mechanical and geometric principles.

While ML may improve engineering models (AI4Science), we can also use scientific computing principles to improve the
performance of ML models for physics-agnostic tasks (Science4AI). Many “black-box” architectures admit alternative
representations from scientific computing, e.g. CNNs as finite differences, multilayer perceptrons as B-splines, ResNets as
discrete differential equations, graph attention networks as finite element/volume methods, or generative models as stochastic
differential equations. We will additionally look to settings where traditional techniques from scientific computing have allowed
improved accuracy and robustness for physics-agnostic tasks in surprising ways. This serves as an attractive basis for designing
machine learning architectures from physical/mathematical principles rather than from ad hoc trial and error.

The course will initially focus on reviewing material necessary for data-driven modeling, including: probability, variational
calculus, and discretizations of partial/ordinary-differential equations. We will consider problems from both engineering settings
and data analytics, focusing on problems of engineering relevance such as inverse problems, reduced-order modeling, and data
assimilation. The course will primarily consist of modules studying a given ML architecture and scientific computing technique in
tandem, culminating in a research project on a topic of relevance to students’ individual research. The course is designed to be
self-contained and accessible to students with students with a background in mechanics/scientific computing but not ML or vice
versa.

# Course Objectives
By the end of this course, you should be able to:

• Comfortably use pytorch or another automatic differentiation library to fit physics to traditional models
* Implement standard schemes (finite differences, volumes, elements) into a simple 1D code
* Use variational principles and numerical analysis to propose novel machine learning architectures
 
# Prerequisites
Formally this course is a sequence with ENM5310. It is ok if you haven't taken that class, but I will assume mathematical maturity, familiarity with python, and a background in probability fundamentals and linear algebra. Expertise with numerical analysis will be beneficial but not assumed.

# Teaching Staff and Office Hours
- Instructor: Dr. Nat Trask <ntrask@seas.upenn.edu>
  - Associate Professor, MEAM
    Office hours: Tuesday 9:15-11 and by Appointment through Ed. 519 Amy Gutmann Hall. 
- Graduate teaching assistant: Cyril Palathinkal <cyrilp@seas.upenn.edu>
  - TA Office hours: TBA
- *Reminder:* All correspondence should be through the ed forum. Emails are provided here for special circumstances (e.g. you can't access the OH building) and will otherwise be ignored.

# Course Website
We will use Canvas (http://canvas.upenn.edu) for assignments, and all material will be hosted on the course github. We will use Ed Discussion for questions and announcements about the course. Ed is accessible through a link on the left panel of our Canvas page. If you have any questions about the class, please create a post! Use email only for sensitive topics. We will do our best to respond to your questions in a timely manner. If you see others’ questions that you can answer or answers that you can improve, do it! Students who have contributed thoughtful comments, questions, and answers throughout the semester will earn extra credit in the class. That said, while hints and suggestions are great contributions, Ed Discussion should not be used for sharing or distributing solutions to any assignments in the class. Our goal is for everyone to understand the material.

For more information on using Ed Discussion, check https://infocanvas.upenn.edu/guides/ed-discussion/

# Course Requirements and Evaluation
Your grade in this class will be determined as a weighted combination of your performance in the
following areas:
- 50% Homework Assignments
  There will be regular assignments consisting of both analysis and programming. You are encouraged to intelligently use LLMs to assist you in writing code, but not in writing up your reports. If you do use LLMs, or any other resources including internet resources or collaboration with other students, you must attribute them to comply with the Penn code of conduct.
- 25% Evaluations
  We will have a few short quizes to ensure understanding of the material. The quizes are designed to force students to study the material, and are not intended to be overly challenging.
- 25% Final Project
  The course will culminate in a project relevant to your research interests, including a short written report and a presentation to the class. Final projects may be done either independently or in groups - consider building collaborations between experimentalists and computational folks. This is a great opportunity to lay the groundwork for a paper that you can finish over the summer!
  
Further details will provided later in the semester.

# Late Policy
One late assignment will be accepted up to two days late. Further late assignments will not be accepted without an excuse from Prof. Trask.

# Collaboration Policy
You are encouraged to discuss the material with your classmates and to work in groups for any homework assignment, but the final product should be your own work. If you collaborate, in any way, you must
acknowledge the collaboration. You should be able to provide a brief explanation of how your learning was improved by the collaboration. If you find this difficult to do, then it is probably the wrong kind of
collaboration. This includes using AI tools or consulting stack overflow.

# University Policies and Resources
This course will be conducted in accordance with all university policies. The university and the School of Engineering & Applied Science also offer numerous resources to students that may be useful. Please let
the instructors know if you have any questions or concerns related to the following:

- Code of Academic Integrity:
  -In accordance with the University’s Code of Academic Integrity (available at https://catalog.upenn.edu/pennbook/code-of-academic-integrity/), all work turned in by students should be an accurate reflection of their knowledge, and, with the exception of working in groups for homework assignments, should be conducted alone. Violation of University Code of Academic Integrity may result in failure of the course.

- Students with Disabilities and Learning Differences
  - Students with disabilities are encouraged to contact Weingarten Learning Resource Center’s Office for Student Disabilities Services for information and assistance with the process of accessing reasonable
accommodations. For more information, visit http://www.vpul.upenn.edu/lrc/sds/, or email lrcmail@pobox.upenn.edu.

- Counseling and Psychological Services (CAPS)
  - CAPS is the counseling center for the University of Pennsylvania. CAPS offers free and confidential services to all Penn undergraduate, graduate, and professional students. For more information, visit http://www.vpul.upenn.edu/caps/.

# ENM5320
AI4Science/Science4AI Spring 2025
MW 10:15am-11:44am in TOWN 309 (1/15 to 4/30)

# Current Schedule
- **Jan 15.**  Quick intro to pytorch and course logistics
  - [Jupyter notebook](https://github.com/natrask/ENM5320/blob/main/Code/ENM_5320_Lec_1.ipynb)
- **Jan 20.** No class for holiday
- **Jan 22.** Finite difference crash course. Implementation, stability analysis, and effective equation.
- **Jan 29.** Extracting finite difference stencils from data. Lax-Wendroff schemes.
- **Jan 31.** Extensions to general point clouds. Generalized finite differences and graph neural networks.

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
    Office hours: TBA
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

# numc

Here's what I did in project 4:

TASK 1:

We began project 4 by naively implementing the matrix methods, but also keeping in mind ideas for later improvements. Ideas we considered, but did not implement during the first stage included allocating contiguous blocks for the matrix data, utilizing repeated squaring for pow, and transposing matrices for mul. These were not implemented during this task, but were considered later. To complete the task itself, we utilized VSCODE Liveshare, with both of us writing in the same file. There was a bit of a learning curve with this, as Enzo had not used it before, but he picked it up and things went smoothly. From there, Kayla debugged, focusing on allocate, allocate_ref, and deallocate in particular, while Enzo moved on to task 2.

For this part, we tried not to optimize immediately, so we could focus on optimizing later during task 4. We believe instead that we should have initially tried some of the optimizations, as it would have lowered the workload when we reached task 4, which ended up taking more time than either of us had planned (TY FOR THE EXTENSION).

TASK 2:

Enzo implemented task 2. This task went pretty smoothly. After reading the documentation and some piazza threads, it was pretty easy to figure out how to use the extension and setup methods. No debugging required for this task :)

TASK 3:

We began this task by splitting the work, with Kayla working on the number and instance methods, while Enzo worked on indexing.

While working on the methods, Kayla utilized piazza, docs, stack overflow and the skeleton code to understand how the python-c interface works. In particular, she found some good examples on stack overflow of how to set error types/messages, and how to work with pyTuples, like those in the get and set methods. With this info, she was able to complete her part of task 3, and begin brainstorming for task 4.

While working on indexing, Enzo utilized similar resources to Kayla. Similar to Kayla, he found sources on pyTuples and pyErrors, but also needed additional information on pySlices. He found the set subscript method to be tricky, but with the help of his resources and posts on piazza, he was able to determine the solution eventually.

Once we each completed our parts, we reconvened to debug. The most common issue we ran into while debugging was issues with casting PyLong, PyDouble, and PyFloat. This resulted in improper values when our matrices were filled with anything other than ints. Additionally, we ran into some issues type-checking, where we forgot to check if certain parameters were of the correct type. We easily fixed this by adding in type checking.

If we were to go back and change how we did this part, we would have shared our resources! After the fact, we discovered that we had used many of the same resources, but had not shared them with each other. This wasted time, as understanding the python-c interface was the hardest part of this task, and finding resources was a time-consuming effort but it made it much simpler.

TASK 4

For this task, we began by making changes to our code, such as removing function calls and changing the order of our for loops in order to decrease the stride. One of these changes that we should have made initially, but did not until much later, was to move variables such as the number of rows or columns from the heap to the stack. Before we moved them, this meant every time we needed the number of rows for a calculation or for the bounds of a for loop, we had to go to the heap, which is time consuming. After moving them, the heap was only accessed one time for each variable, and then every reference after that occurred on the much-faster stack. Another change we came to late was to reuse calculations. Although it seems small, repeated calculations can add significantly to the time.

Perhaps the most important code changes, however, were the ones that we mentioned above. Initially, in our naive implementation, we allocated the individual rows of data in our matrix separately. This meant that on the heap, the memory was broken into chunks, and could be very far apart. In turn, this makes accesses very slow, with no locality. By altering this, and implementing the matrix data as a 1-d array, where all of the space was allocated at once, our data became localized. This meant that lookup was in one place, and much faster for repeated lookups, rather than moving all across the heap to access the different chunks. Additionally, our pow function was very slow, because it was performing many many matrix multiplications. By implementing repeated squaring, we greatly decreased the number of matrix multiplications. This in turn greatly sped up pow.

Next, we put in SIMD instructions. We used SIMD instructions more sparsely in our efforts to speed up our code, but they were a very important part of speeding up our simple matrix calculations, such as add and sub. With the SIMD add and store, we were able to optimize our code to perform multiple additions in parallel.

Lastly, we utilized OMP and parallel for loops to speed up almost every avenue of our code. For loops were used everywhere, from allocation to addition to pow, so by parallelizing them, we were able to speed up almost everything.

OVERALL:

This project was a lot of work, mostly in the later tasks. That being said, we learned a lot while trying to meet the perfomance requirements in the later tasks. We tried many different versions of mul_matrix to achieve the optimal speedup. First, we tried transposing the matrix, then cache-block transposing, then using a different algorithm. None of these worked to achieve the speedup needed, and were in fact too complicated. We found that the most speedup occurred simply by moving the correct values to the stack, placing the for loops in the order with minimum stride, and implementing repeated squaring. While our solution was not perfect, We believe we learned the most through the different testing and implementations that we tried.

Some things that surprised us with this project were: how much faster the stack was than the heap, and how much the order of the for loop matters for cache access. While we knew the stack was faster than the heap, it was hard to quanitfy since both were very fast. When looking at something on a very large scale like on this project, it became easier to understand, as those little differences in time were magnified. Additionally, we did not realize the power of cache locality, and the time it took for cache misses. This meant we were very surprised how much of a difference it made when we minimized cache misses.

# def count 7 - easy difficulty 
# bubble sort intuition?
# tower of hanoi 
def count_7(n):
    if(n//10==0):
        if n%10 == 7:
            return 1
        return 0
    if n%10 == 7: return 1 + count_7(n//10)
    return 0 +  count_7(n//10)



# bubble sort implementation

def bubble_sort(a):
    n = len(a) # returns the number of items in a container. How generic 
    for i in range(n):
        for j in range(n-1-i):
            if (a[j]>a[j+1]):
                temp = a[j+1]
                a[j+1] = a[j]
                a[j] = temp
    return a

 


# tower of hanoi. Explicitly moving it from A to C. Moving two plates is just moving one plate thrice. Moving 3 plates is just moving 2 plates, then moving the 3rd plate, then moving the 2 plates again. All 3 operations. One pole to the other, use any of the 3 poles 
def towerofhanoi(from_rod, side_rod, to_rod, n):
    if n==1:
        print("Plate shifted from " + from_rod + " to " + to_rod)
         
    else:
        towerofhanoi(from_rod,to_rod,side_rod,n-1)
        print("Plate shifted from " + from_rod + " to " + to_rod)
        towerofhanoi(side_rod,from_rod,to_rod,n-1)
towerofhanoi("A","B","C",3)

    
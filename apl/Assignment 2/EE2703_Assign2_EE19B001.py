"""
                             EE2703 Applied Programming Lab - 2019
                                 Assignment 1: Solution
                             Arun Krishna A M S - EE19B001
                                  24th February 2021
"""

from sys import argv, exit
import numpy as np;
from numpy import pi,exp,inf;

"""
Checking whether the given inputs are the required ones, and whether the given file name 
is correct and the file exists. If not the open function throws an error.
"""

if len(argv) == 2 :       
       filename = argv[1];
else:
       print("Number of arguments must be 2");
       exit(0);
try:
       f = open(filename);
       lines = f.readlines();
       f.close();
except:
       print("File not found");
       exit(0);


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
              
"""
The constant variables 'startCircuit' and 'endCircuit' determines the start and end of the
described circuit.  The 'start' and 'end' variables denote thet starting and ending line of 
the circuit simulation program. 

findAC list stores all possible AC independent sources. While findfreq list stores the
corresponding frequenciesof those sources.
"""


startCircuit = '.circuit';
endCircuit = '.end';
start = -1;
end = -2;
AC = '.ac';
findAC = [];
findfreq = [];


"""
The location of each line is stored, and the endline character is removed from each 'line'.
Following that, the 'line' is split into two parts, the comment and the program code. The 
comment is ignored and only the program code is saved in the variable 'line'. The tab spaces
are converted to spaces and then the lines are stripped out of their leading and trailing 
spaces and then compared to find the beginning and the ending of the code.
"""


for line in lines:

       location = lines.index(line);
       line = line.replace('\n','');
       line = line.split('#')[0];
       line = line.replace('\t',' ');       
       line=line.strip();
       lines[location] = line;

       if line[:len(startCircuit)] == startCircuit:
              start = location;
       elif line[:len(endCircuit)] == endCircuit:
              end = location;       
       elif line[:len(AC)] == AC:
              line = line.split(" ");
              line = [word for word in line if word != ""];
              try:
                     line[2] = float(line[2])
              except:
                     print("Numeric Values Only: Line ",location+1);
                     exit(0);
              findAC.append(line[1]);
              findfreq.append(line[2]);
              
       
if start >= end:
       print("Circuit Block Invalid");
       exit(0);


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
class 'Element' has been created which stores whether the element is a DC/AC, the element type, its value, frequency
and phase.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

class Element:       
       def __init__(self,checkAC,ElementType,node1,node2,node3,node4,value,freq,phase):
              self.checkAC = checkAC;
              self.ElementType = ElementType;
              self.node1 = node1;
              self.node2 = node2;
              self.node3 = node3;
              self.node4 = node4;
              self.value = value;
              self.freq = freq;
              self.phase = phase;
       def __repr__(self):
              string = "\ncheckAC " + str(self.checkAC) + " Element " + str(self.ElementType) + " node1 " + str(self.node1) +  " node2 " + str(self.node2)
              string = string + "\n node3 " + str(self.node3) + " node4 " + str(self.node4) + " value " + str(self.value) + " freq " + str(self.freq) + " phase " + str(self.phase);
              return string;
       
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

circuitElements = [];
Token = [];
elementName = [];

for line in lines[start+1:end]:
       """
       The location of each line is stored. From 'line' a 'linelist' of words/tokens are 
       created i.e., tokens are separated by space are stored in 'linelist'. If the linelist 
       is empty this iteration is ignored
       """
       location = lines.index(line);
       linelist = line.split(" ");
       linelist = [elem for elem in linelist if elem != ""];

       """
       Following that, the 'linelist' is checked for whether it is a resistor, capacitor, inductor
       dependent and independent voltage and currect sources and appended to the 'Token'. In case of
       resistor, inductor, capacitor independent voltage and current source, the relations depends 
       on only two nodes - thus 4 tokens. In the case of voltage dependent sources, It is not currently 
       supported. The intial letter of label of the elements denote the element type. The label of nodes 
       must also be alphanumeric. 
       """

       if linelist == []:
              continue;
       
       elementName.append(linelist[0][0] + str(location - (start+1)));
              
       if linelist[0][0] == 'R' or linelist[0][0] == 'L' or linelist[0][0] == 'C':       
              if len(linelist) != 4:
                     print("Incorrect Number of Parameters: Line ",location+1);
                     exit(0)
              if linelist[1].isalnum() != True or linelist[2].isalnum() != True :
                     print("Incorrect Node Designation - only alphanumeric variables: Line ",location+1);
                     exit(0);
              try:
                     linelist[3]=float(linelist[3])
              except:
                     print("Numeric Values Only: Line ",location+1);
                     exit(0);
              elem = Element(0,linelist[0],linelist[1],linelist[2],'','',linelist[3],'','');       
              
              
       elif linelist[0][0] ==  'E' or linelist[0][0] == 'G' or linelist[0][0] ==  'H' or linelist[0][0] == 'F':
              print("Currently dependent sources are not supported");
              exit(0);
              


       elif linelist[0][0] == 'V' or linelist[0][0] == 'I' :        
              if linelist[1].isalnum() != True or linelist[2].isalnum() != True :
                     print("Incorrect Node Designation - only alphanumeric variables: Line ",location+1);
                     exit(0);
        
              #In case of AC independent sources, the number of tokens = 6 is checked and whether the frequency for the
              #element is given or not. The phase and the value must be of float value
              if len(linelist) == 6 and linelist[3] == "ac":
                     try:
                            ACindex = findAC.index(linelist[0]);
                            frequency = findfreq[ACindex];
                            try:
                                   linelist[4] = float(linelist[4]);
                                   linelist[5] = float(linelist[5]);
                                   elem = Element(1,linelist[0],linelist[1],linelist[2],'','',linelist[4],frequency,linelist[5]);
                            except:
                                   print("Numeric Values Only: Line ",location+1);
                                   exit(0);                                  
                     except:
                            print("Frequency not found: Line", location+1);

              #In case of DC independent sources, the number of tokens = 5 is checked. The phase and the value must be of float value
              elif len(linelist) == 5 and linelist[3] == "dc":
                     try:
                            linelist[4] = float(linelist[4]);
                            elem = Element(0,linelist[0],linelist[1],linelist[2],'','',linelist[4],'','');
                     except:
                            print("Numeric Values Only: Line ",location+1);
                            exit(0);                                  
              else:
                     print("Incorrect Number of Parameters: Line ",location+1);
                     exit(0);
                     
                     
       else:
              print("Invalid Element: Line ",location+1);
       
       circuitElements.append(elem);

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Initially the circuit block is checked whether it has AC components or not. If exists the frequency is 
stored and the values of Capacitors, Inductors and Independent sources are changed to their reactance 
i.e., complex values. In case of DC the frequency is set to 10 to the power of -50, which essentially for
all practical purposes is a DC. 
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


checkAC = 0;
freq = 1e-50;
for elem in circuitElements:
       if elem.checkAC == 1:
              checkAC = 1;
              freq = elem.freq;
              break;  
              
for elem in circuitElements:
       if elem.freq != freq and elem.freq != '':
              print("Currently Sources with the same frequencies are allowed. Come back next time");
              exit(0);
       #X_L = j*L*2pi*f        
       if elem.ElementType[0] == 'L':
              elem.value = 1j*elem.value*freq*2*pi; 
       #X_C = 1/(j*C*2pi*f)
       elif elem.ElementType[0] == 'C':
              if checkAC == 1:
                     elem.value = 1/(1j*elem.value*freq*2*pi); 
              elif checkAC == 0:
                     elem.value = inf; 
       #Including phase in case of the independent sources.
       elif (elem.checkAC == 1 and (elem.ElementType[0] == 'V' or elem.ElementType[0] == 'I')):
              elem.value = elem.value/2*exp(1j*elem.phase*pi/180); 


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Corresponding Stamps for each element is created, and the stamps are then added to the matrix 'M'
which is then inversed and multiplied with the matrix 'b'

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


nodelist = [];
currentlist = [];

#If the element is a voltage source, then auxillary currernts have to be defined. Thus they are added in currentlist
for elem in elementName:
       if elem[0] == 'V':
                     currentlist.append(elem);

#All the nodes of all the elements are taken, and the duplicate nodes are removed i.e., the array nodelist contains
#distinct nodes
for elem in circuitElements:
       nodelist.append(elem.node1);
       nodelist.append(elem.node2);
       nodelist.append(elem.node3);
       nodelist.append(elem.node4);

temp = [] 
for i in nodelist: 
       if i not in temp: 
              temp.append(i);
nodelist = temp;

#Remove the '' element from the nodelist 
nodelist = [elem for elem in nodelist if elem != ''];

#creating a matrixlist that would help us in calculation of matrix M and b
matrixlist = nodelist + currentlist;

#Creation of Matrices M and b
sizeM = len(matrixlist);
M = np.zeros((sizeM,sizeM), dtype = 'complex_');
b = np.zeros((sizeM,1),dtype = 'complex_');

#For all the elements stamps are created, and then added to the matrix M and b
for elem in circuitElements:
       
       location = circuitElements.index(elem);
       if elem.ElementType[0] == 'R' or elem.ElementType[0] == 'C'or elem.ElementType[0] == 'L': 
              Gm = 1/elem.value;
              nodek = matrixlist.index(elem.node1);
              nodel = matrixlist.index(elem.node2);
              M[nodek,nodek] = M[nodek,nodek] + Gm; 
              M[nodel,nodel] = M[nodel,nodel] + Gm;
              M[nodek,nodel] = M[nodek,nodel] - Gm; 
              M[nodel,nodek] = M[nodel,nodek] - Gm; 
       
       elif elem.ElementType[0] == 'I':
              nodek = matrixlist.index(elem.node1);
              nodel = matrixlist.index(elem.node2);
              b[nodek] = b[nodek] + elem.value;
              b[nodel] = b[nodel] - elem.value;
       
       elif elem.ElementType[0] == 'V':
              nodek = matrixlist.index(elem.node1);
              nodel = matrixlist.index(elem.node2);
              nodeiKL = matrixlist.index(elementName[location]);          
              M[nodek,nodeiKL] = M[nodek,nodeiKL] + 1; 
              M[nodeiKL,nodek] = M[nodeiKL,nodek] + 1;
              M[nodel,nodeiKL] = M[nodel,nodeiKL] - 1; 
              M[nodeiKL,nodel] = M[nodeiKL,nodel] - 1; 
              b[nodeiKL] = b[nodeiKL] + elem.value;
              

#If no ground is given, any one node is taken as the Ground, and the values are calculated with respect
#to it. The chose node is then printed
possibleGND = "GND";
if "GND" not in nodelist:
       possibleGND = nodelist[0];
       print(nodelist[0],"is chosen as ground");


#The column and row corresponding to the GND is deleted from M,b and matrixlist
#GND point is removed from matrixlist since it is used for finding the index for printing the node voltages
#and auxillary currents
GNDindex = matrixlist.index(possibleGND);
M = np.delete(M,GNDindex,0);
M = np.delete(M,GNDindex,1);
b = np.delete(b,GNDindex,0);
matrixlist.remove(possibleGND)
nodelist.remove(possibleGND)

#Solving for X
X = np.linalg.solve(M,b)

print("Voltage at node ",possibleGND," is ", 0);


for i in range(len(b)):
       if i < len(nodelist):
              print("Voltage at node ",nodelist[i]," is ", X[i]);
       else:
              temp = elementName.index(matrixlist[i]);
              elem = circuitElements[temp]; 
              print("Current through ",elem.ElementType ," is ", X[i]);























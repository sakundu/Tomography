import sys
import os

mem_name = sys.argv[1]
mem_inst = sys.argv[2]
lef_file = sys.argv[3]

mem_input = []
mem_output = []
isPin = 0
curPin = ""

with open(lef_file) as f:
    contents = f.read().splitlines()
f.close()

for line in contents:
    items = line.split()
    if(len(items) == 2 and items[0] == "PIN"):
        isPin = 1
        curPin = items[1]
    elif(len(items) == 2 and items[0] == "END" and items[1] == curPin):
        isPin = 0
        curPin = ""
    
    if(isPin == 1 and items[0] == "DIRECTION" and items[1] == "INPUT"):
        if(curPin != "clk"):
            mem_input.append(curPin)
    elif(isPin == 1 and items[0] == "DIRECTION" and items[1] == "OUTPUT"):
        mem_output.append(curPin)

inputMemCnt = len(mem_input)
outputMemCnt = len(mem_output)

with open("tomo_train.v") as inFile:
    inNetlist = inFile.read().splitlines()
inFile.close()

outFile = open("tomo_train_wMacro.v", 'w')

write = 1
inSigNet = []
outSigNet = []
firstWire = 1
for line in inNetlist:
    items = line.split()
    if(len(items) == 2 and items[0] == "input"):
        if(outputMemCnt > 0):
            sig = items[1].split(',')
            inSigNet.append(sig[0])
            outputMemCnt = outputMemCnt - 1
            write = 0
        else:
            write = 1

    if(len(items) == 2 and items[0] == "output"):
        if(inputMemCnt > 0):
            sig = items[1].split(',')
            outSigNet.append(sig[0])
            inputMemCnt = inputMemCnt - 1
            write = 0
        else:
            write = 1

    if(line == "endmodule"):
        outFile.write(str(mem_name) + " " + str(mem_inst) + " (" + '\n')
        pinCnt = 0
        pinArray = 0
        arrayPin =""
        for i in range(len(mem_input)):
            pinComp = mem_input[i].split('[')
            if(len(pinComp) > 1):
                if(pinArray == 0):
                    arrayPin = pinComp[0]
                    outFile.write("." + str(arrayPin) + "({" + str(outSigNet[i]))
                    pinArray = 1
                elif(pinComp[0] == arrayPin):
                    if(i == len(mem_input) - 1):
                        outFile.write(", " + str(outSigNet[i]) + "}), ")
                    else:
                        outFile.write(", " + str(outSigNet[i]))
                elif(pinComp[0] != arrayPin):
                    arrayPin = pinComp[0]
                    outFile.write("}), " + '\n' + "." + str(arrayPin) + "({" + str(outSigNet[i]))
            else:
                if(pinArray == 1):
                    outFile.write("}), ." + str(mem_input[i]) + "(" + str(outSigNet[i]) + "), ")
                    pinArray = 0
                else:
                    outFile.write("." + str(mem_input[i]) + "(" + str(outSigNet[i]) + "), ")

            pinCnt = pinCnt + 1
            if(pinCnt % 10 == 0):
                outFile.write('\n')
        
        pinArray = 0

        for i in range(len(mem_output)):
            pinComp = mem_output[i].split('[')
            if(len(pinComp) > 1):
                if(pinArray == 0):
                    arrayPin = pinComp[0]
                    outFile.write("." + str(arrayPin) + "({" + str(inSigNet[i]))
                    pinArray = 1
                elif(pinComp[0] == arrayPin):
                    if(i == len(mem_output) - 1):
                        outFile.write(", " + str(inSigNet[i]) + "}), ")
                    else:
                        outFile.write(", " + str(inSigNet[i]))
                elif(pinComp[0] != arrayPin):
                    arrayPin = pinComp[0]
                    outFile.write("}), " + '\n' + "." + str(arrayPin) + "({" + str(inSigNet[i]))
            else:
                if(pinArray == 1):
                    outFile.write("}), ." + str(mem_output[i]) + "(" + str(inSigNet[i]) + "), ")
                    pinArray = 0
                else:
                    outFile.write("." + str(mem_output[i]) + "(" + str(inSigNet[i]) + "), ")

            pinCnt = pinCnt + 1
            if(pinCnt % 10 == 0):
                outFile.write('\n')

        outFile.write(".clk(clk) );" + '\n\n')

    if(write==1):
        outFile.write(line + '\n')

os.system(f"mv tomo_train.v tomo_train_woMacro.v")
os.system(f"mv tomo_train_wMacro.v tomo_train.v")



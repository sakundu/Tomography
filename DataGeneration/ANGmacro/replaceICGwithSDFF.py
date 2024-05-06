import sys
import os

with open("ca53_cpu.v") as inFile:
    inNetlist = inFile.read().splitlines()
inFile.close()

outFile = open("ca53_cpu_wMem.v", 'w')

write = 1
inSigNet1 = ""
inSigNet2 = ""
outSigNet = ""
cellName = ""
cellUnit = ""
pinSE = 0
pinENA = 0

for line in inNetlist:
    items = line.split()
    if (len(items) > 1 and items[0].startswith("CLKGATE")):
        #cellComp = items[0].split('_')
        if items[0] == "CLKGATETST_X1":
            cellName = "SDFFS_X1"
            cellUnit = items[1]
        elif items[0] == "CLKGATETST_X2":
            cellName = "SDFFS_X2"
            cellUnit = items[1]
        elif items[0] == "CLKGATETST_X4":
            cellName = "SDFFS_X2"
            cellUnit = items[1]
        elif items[0] == "CLKGATETST_X8":
            cellName = "SDFFS_X2"
            cellUnit = items[1]
        elif items[0] == "CLKGATE_X1":
            cellName = "DFF_X1"
            cellUnit = items[1]
        elif items[0] == "CLKGATE_X2":
            cellName = "DFF_X2"
            cellUnit = items[1]
        elif items[0] == "CLKGATE_X4":
            cellName = "DFF_X2"
            cellUnit = items[1]
        elif items[0] == "CLKGATE_X8":
            cellName = "DFF_X2"
            cellUnit = items[1]
        write = 0
    elif (write == 0 and items[0].startswith(".E")):
        pinComp = line.split('(')
        inSigNet1 = pinComp[1]
        pinENA = 1
    elif (write == 0 and items[0].startswith(".SE")):
        pinComp = line.split('(')
        inSigNet2 = pinComp[1]
        pinSE = 1
    elif (write == 0 and items[0].startswith(".GCK")):
        pinComp = line.split('(')
        outSigNet = pinComp[1]
    elif (write == 0 and items[0].startswith(".CK")):
        continue
    elif (write ==0 and line == ");" and pinSE == 1 and pinENA == 1):
        outFile.write(cellName + " " + cellUnit + '\n')
        outFile.write(".D(" + inSigNet1 + '\n')
        outFile.write(".SE(" + inSigNet2 + '\n')
        outFile.write(".SI(" + inSigNet1 + '\n')
        outFile.write(".SN(" + inSigNet2 + '\n')
        outFile.write(".CK(clk)" + '\n')
        outFile.write(".Q(" + outSigNet + '\n')
        outFile.write(".QN()" + '\n')
        outFile.write(");" + '\n')
        write = 1
        pinSE = 0
    elif (write ==0 and line == ");" and pinENA == 1):
        outFile.write(cellName + " " + cellUnit + '\n')
        outFile.write(".D(" + inSigNet1 + '\n')
        outFile.write(".CK(clk)" + '\n')
        outFile.write(".Q(" + outSigNet + '\n')
        outFile.write(".QN()" + '\n')
        outFile.write(");" + '\n')
        write = 1
        pinSE = 0
    else:
        outFile.write(line + '\n')

os.system(f"mv ca53_cpu.v ca53_cpu_ORG.v")
os.system(f"mv ca53_cpu_wMem.v ca53_cpu.v")



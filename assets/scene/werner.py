import json
# Werner BOBBLE-HEAD: tall head (6px) with the proper 2x2 white eyes back, inset with cheeks.
# Body/book/feet proportions unchanged (4px) — just a bigger head.
W,H=12,12
PAL={
 ".":None,"A":[226,226,220],"K":[234,186,154],
 "J":[48,124,88],"y":[96,164,124],"n":[46,44,52],
 "E":[248,248,246],"x":[26,20,18],
 "P":[232,226,206],"i":[120,82,48],"1":[156,60,52],
}
g=[["." for _ in range(W)] for _ in range(H)]
def rect(x0,y0,x1,y1,c):
    for y in range(y0,y1+1):
        for x in range(x0,x1+1):
            if 0<=x<W and 0<=y<H: g[y][x]=c
def px(x,y,c):
    if 0<=x<W and 0<=y<H: g[y][x]=c

# head — tall bobble (x2-9, 6px), 2x2 white eyes inset with cheeks
rect(2,0,9,1,"K"); px(2,0,"A"); px(9,0,"A"); px(2,1,"A"); px(9,1,"A")   # crown + side hair
rect(2,2,9,2,"K")                          # forehead
rect(2,3,9,4,"K")                          # eyes row + cheeks
rect(3,3,4,4,"E"); rect(7,3,8,4,"E")       # 2x2 white eyes
px(4,4,"x"); px(7,4,"x")                   # pupils
px(2,3,"A"); px(9,3,"A")                   # sideburns
rect(3,5,8,5,"K")                          # jaw (tapered)
# body — UNCHANGED proportions (shoulders / book / feet), centered under the head
rect(3,6,8,6,"J"); px(5,6,"y"); px(6,6,"y")               # shoulders + collar
px(2,7,"K"); rect(3,7,8,7,"P"); px(5,7,"i"); px(6,7,"i"); px(9,7,"K")   # hands + open book
rect(3,8,8,8,"1")                          # red cover
px(3,9,"n"); px(4,9,"n"); px(7,9,"n"); px(8,9,"n")        # feet
json.dump({"palette":PAL,"rows":["".join(r) for r in g]},open("/tmp/werner.json","w"))
print("wrote bobble-head werner")

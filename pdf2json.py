import numpy as np
from PIL import Image
import PIL
import cv2
from pdf2image import convert_from_path
import math
import string
import json as j
import io
from google.cloud import vision_v1
import sys

def toarr(image):
    if isinstance(image,PIL.PpmImagePlugin.PpmImageFile):
        image = np.array(image)
    r,c,_ = image.shape
    return image[:,int(c*0.07058823):int(c*0.92352941)]
    

def boundary(image):
    image = toarr(image)
    image1 = np.where(image < 50 , 0, 255)
    
    
    r,c,_ = image.shape
    dd = BreakPoints(image1[int(r*0.77):,:],0,x = int(c/2))

    ll = [ dd[i+1] - dd[i] for i in range(len( dd )-1)  ]
    image = image[:dd[ll.index(max(ll))]+int(r*0.77)+30,:,:]
    r,c,_ = image.shape
    image = cv2.line(image,(0,0),(0,r),(255,255,255),30)
    image = cv2.line(image,(0,0),(c,0),(255,255,255),30)
    image = cv2.line(image,(0,r),(c,r),(255,255,255),30)
    image = cv2.line(image,(c,0),(c,r),(255,255,255),30)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 160,255,cv2.THRESH_BINARY_INV)[1]
    gray = np.float32(thresh)
    dst = cv2.cornerHarris(gray,2,3,0.04)
    indices = np.argwhere((dst>0.01*dst.max()) == True)
    indlist = indices.tolist()
    distances = np.linalg.norm(indices,axis = 1)
    sorted_distances = sorted(distances)
    start = indices[np.where(distances == sorted_distances[0] )[0][0],:]
    end = indices[np.where(distances == sorted_distances[-1] )[0][0],:]
    image = image[start[0]-10:end[0]+10,start[1]-10:end[1]+10]
    return image
          
def detect(sec,path,breaks = None,cut = None):
    file_path = path+".tiff"
    r,c,_ = sec.shape
    if cut is not None:
        cutf, cuts = cut
    else:
        cutf = 0
        cuts = 0
        
    if breaks is not None:
        for x in range(1,len(breaks)-2):
            cv2.line(sec,(cutf,breaks[x]),(cutf,breaks[x+1]-int(r*0.080801551)),(0,0,0),6)
            cv2.line(sec,(cuts,breaks[x]),(cuts,breaks[x+1]-int(r*0.080801551)),(0,0,0),6)
        cv2.line(sec,(cutf,breaks[-2]),(cutf,breaks[-1]),(0,0,0),6)
        cv2.line(sec,(cuts,breaks[-2]),(cuts,breaks[-1]),(0,0,0),6)
        cv2.imwrite(file_path,sec)
        client = vision_v1.ImageAnnotatorClient()
    mime_type = "image/tiff"
    with io.open(file_path, "rb") as f:
        content = f.read()
    input_config = {"mime_type": mime_type, "content": content}
    features = [{"type_": vision_v1.Feature.Type.DOCUMENT_TEXT_DETECTION}]

    pages = [1, 2, -1]
    requests = [{"input_config": input_config, "features": features, "pages": pages}]

    response = client.batch_annotate_files(requests=requests)
    LIST = []
    for image_response in response.responses[0].responses:
        for page in image_response.full_text_annotation.pages:
            for block in page.blocks:
                for par in block.paragraphs:
                    for word in par.words:
                        wor = []
                        for symbol in word.symbols:
                            wor.append(symbol.text)
                        LIST.append( ( [[v.x,v.y] for v in word.bounding_box.vertices],"".join(wor)) )
    return LIST


def blockIsIn(block,v):
    for x in block:
        if not isIn(x,v):
            return False
    return True

def isIn(p,v):
    return p[0]>v[0][0]-5 and p[0]<v[1][0]+5 and p[1]>v[0][1]-5 and p[1]<v[1][1]+5

def BreakPoints(a,adjust,x,total = None):
    r,c,_ = a.shape
    breakpoints = []
    breakpoints.append(0)
    change = False
    for row in range(0,a.shape[0]):
        if a[row,x,0] == 255:
            change = True
            
        if change and a[row,x,0] == 0:
            breakpoints.append(row-adjust)
            change = False
            if total is not None and len(breakpoints) == total:
                break
    return breakpoints   


def segmentation(a,d):
    r,c,_ = a.shape
    img = a.copy()
    img = np.where(img < 220, 0,img)
    img = cv2.blur(img , (10,3))
    img = np.where(img < 200 , 0,255)
    img = np.float32(img)
    l = []
    for x in range(0,img[:,1200,0].shape[0]-60):
        l.append(( (img[x:x+d,1200,0]/255).sum() , x))
    sort = sorted(l,key = lambda x: x[0],reverse = False)
    pts = [0]
    def toadd(pts,pt):
        for x in pts:
            if abs(pt-x) < 100:
                return False
        return True
    for (s,x) in sort:
        if toadd(pts,x) and s < 10:
            pts.append(x)
    breakpoints = sorted(pts)
    breakpoints.append(r)
    sections = []
    for ind in range(len(breakpoints)-1):
        prevp = breakpoints[ind]
        nextp = breakpoints[ind+1]
        if abs(prevp-nextp) > 100:
            sections.append(a[prevp:nextp,0:c])
    return sections,breakpoints




def draw(img,cords):
    r,c,_ = img.shape
    for (a,b) in cords:        
        img = cv2.circle(img,a,5,(255,0,0),-1)
        img = cv2.circle(img,b,5,(0,0,0),-1)
    plt.imshow(img)

def extractsec1(LIST,c,z):
    RATINGLIST = []
    STATEL = []
    STATE = [(int(0.76*c), int(0.376*z)), (int(0.883*c), int(0.549*z))]
    RISKID = [(int(0.88*c), int(0.173*z)), (int(1*c),int(0.35*z))]
    RATINGEFFECTIVE = [(int(0.279*c), int(0.387*z)), (int(0.371*c), int(0.56*z))]
    PRODUCTION = [(int(0.565*c),int(0.383*z)),(int(0.663*c),int(0.564*z))]
    RISKNAME = [(int(0.193*c), int(0.214*z)), (int(0.632*c),int(0.346*z))]
    STATECODE = [(int(0*c),int(0.719*z)),(int(0.1800*c),int(0.86*z))]
    CARRIER = [(int(0.07*c),int(0.88*z)),(int(0.135*c),int(1*z))]
    EFF_DATE = [(int(0.553*c),int(0.88*z)),(int(0.651*c),int(1*z))]
    EXP_DATE = [(int(0.803*c),int(0.88*z)),(int(1*c),int(1*z))]
    POLICYNO = [(int(0.26*c),int(0.88*z)),(int(0.43*c),int(1*z))]
    sec1list = [STATE,RISKID,RATINGEFFECTIVE,PRODUCTION,RISKNAME,STATECODE,CARRIER,EFF_DATE,EXP_DATE,POLICYNO]

    RISKLIST = []
    risk_name = ""
    risk_id = ""
    rating_effective_date = ""
    production_date = ""
    state = ""
    state_code = ""
    carrier = ""
    eff = ""
    exp = ""
    pol = ""
    for ind,x in enumerate(LIST):
        if blockIsIn(x[0],RISKID):
            risk_id = x[1]
        if blockIsIn(x[0],RATINGEFFECTIVE):
            RATINGLIST.append(x[1])
        if blockIsIn(x[0],RISKNAME):
            RISKLIST.append(x[1])
        if blockIsIn(x[0],PRODUCTION):
            production_date = x[1]
        if blockIsIn(x[0],STATE):
            state = x[1]            
        if blockIsIn(x[0],STATECODE):
            STATEL.append(x[1])
        if blockIsIn(x[0],CARRIER):
            carrier = x[1]
        if blockIsIn(x[0],EFF_DATE):
            eff = x[1]
        if blockIsIn(x[0],EXP_DATE):
            exp = x[1]
        if blockIsIn(x[0],POLICYNO):
            pol = x[1]
        
    rating_effective_date = [x for x in RATINGLIST if ("/" in x)][0]
    risk_name =  " ".join(RISKLIST)
    state_code = " ".join(STATEL)

    if risk_name == "":
        print("Risk Name Not Found")
    if risk_id == "":
        print("Risk ID Not Found")
    if rating_effective_date == "":
        print("Rating Effective Not Found")
    if production_date == "":
        print("Production Date Not Found")
    if state == "":
        print("State Not Found")
    if state_code == "":
        print("State Code Not Found")
    if carrier == "":
        print("Carrier Not Found")
    if eff == "":
        print("Eff Date Not Found")
    if exp == "":
        print("Exp Date Not Found")
    if pol == "":
        print("Policy Number Not Found")
        
    temp = {
                            "risk_name": cleant(risk_name),
                            "risk_id": risk_id,
                            "rating_effective_date": rating_effective_date,
                            "production_date": production_date,
                            "state": state,
                            "carrier": carrier,
                            "policy_no": pol,
                            "eff_date": eff,
                            "exp_date": exp,
                            "code": "",
                            "elr": "",
                            "dratio": "",
                            "payroll": "",
                            "expected_losses": "",
                            "exp_prim_losses": "",
                            "claim_data": "",
                            "ij": "",
                            "of": "",
                            "act_inc_losses": "",
                            "act_prim_losses": "",
                            "statecode": state_code,
                            "Policy Total": "",
                            "Subject Premium": "",
                            "Total Act Inc Losses": ""
                        }
    return temp

def extractsec3(x,data):
    exsec2 = data.copy()
    claim_data = x[6]
    code = x[0]
    try:
        ij = int(x[7])
    except:
        ij = ""
    of = x[8] if len(x[8]) == 1 else ""
    try:
        act_inc_losses = x[9].replace(".",",").replace(",","")
    except:
        act_inc_losses = ""

    try:
        act_prim_losses = int(x[10].replace(".",",").replace(",",""))
    except:
        act_prim_losses = ""

    try:
        payroll = int(x[3].replace(",",""))
    except:
        payroll = ""

    try:
        elr = float(x[1].replace(" ","").replace(",","."))
    except:
        elr = x[1]
    try:
        expected_losses = int(x[4].replace(",",""))
    except:
        expected_losses = ""  

    try:
        dratio = float("."+(x[2].replace(" ","").replace(",","").replace(".","")))
    except:
        dratio = ""
    try:
        exp_prim_losses = int(x[5].replace(",",""))
    except:
        exp_prim_losses = "" 
    exsec2["claim_data"] = claim_data.translate(str.maketrans('', '', '!"#%&\'()*+,-./:;<=>?@[\\]^_`{|}~'))
    exsec2["code"] = code
    exsec2["ij"] = str(ij)
    exsec2["of"] = of
    exsec2["act_inc_losses"] = str(act_inc_losses)
    exsec2["act_prim_losses"] = str(act_prim_losses)
    exsec2["payroll"] = str(payroll)
    exsec2["elr"] = str(elr)
    exsec2["expected_losses"] = expected_losses
    exsec2["dratio"] = str(dratio)
    exsec2["exp_prim_losses"] = exp_prim_losses
    exsec2["claim_data"] = claim_data
        
    return exsec2

def same(P,a,b):
    p = P.copy()
    r , c ,_= p.shape
    rm = 50
    wid = 20
    if c > 2500:
        rm = 100
        print("here")
    p = np.where(p < 170 , 0,255)
    a = cv2.blur(p , (a,1))
    a = np.where(a > rm , 255,0)
    
    a = cv2.line(np.float32(a), (0,0), (c, 0), (0, 0, 0), thickness=20)
    hlist = [BreakPoints(a,0,x) for x in range(0,c,10)]
    h = sorted(hlist, key = lambda x: len(x), reverse = True)[0]
    h.append(r)
    h = hcheck(h)
    a = cv2.blur(p , (1,b))
    a = np.where(a > rm , 255,0)
    line_thickness = 2
    for x in h:
        a = cv2.line(np.float32(a), (0, x), (c, x), (0, 0, 0), thickness=line_thickness)
    a = cv2.line(a, (0, 0), (0, r), (0, 0, 0), thickness=10)    
    a = cv2.line(a, (c,0), (c, r), (0, 0, 0), thickness=10)
    
    yy = []
    cords = []
    y = 0
    ignore_first = False
    for ind in range(1,len(h)):
        d = (h[ind] - h[ind-1])
        mid0 = int(h[ind-1] + (d/2) )
        mid1 = int(h[ind-1] + (d/4) )
        mid2 = int(h[ind-1] + (3*d/4) )
        sprev = 0
        MID = None
        for mid in [mid0,mid1,mid2]:
            s = 0
            temp_cords = []
            temp_yy = []
            for x in range(c):
                if a[mid,x].sum() == 0 and ignore_first and abs(y-x) > wid and abs(h[ind] - h[ind-1]) > 20:
                    if x < y:
                        y = x
                        continue
                    temp_cords.append([(y,h[ind-1]),(x,h[ind])])
                    name = np.uint8(P[h[ind-1]:h[ind],y:x,:])
                    temp_yy.append(name)
                    y = x
                elif a[mid,x].sum() == 0:
                    ignore_first = True
            if len(temp_yy) !=0 and len(temp_cords) != 0:
                s = s+len(temp_yy)
            if s > sprev:
                MID = mid
                sprev = s
                
        temp_cords = []
        temp_yy = []
        ignore_first = False
        for x in range(c):
            if a[MID,x].sum() == 0 and ignore_first and abs(y-x) > wid and abs(h[ind] - h[ind-1]) > 20:
                if x < y:
                    y = x
                    continue
                temp_cords.append([(y,h[ind-1]),(x,h[ind])])
                name = np.uint8(P[h[ind-1]:h[ind],y:x,:])
                temp_yy.append(name)
                y = x
            elif a[MID,x].sum() == 0:
                ignore_first = True
        if len(temp_yy) !=0 and len(temp_cords) != 0:
            yy.append(temp_yy)
            cords.append(temp_cords)
    
    return yy,cords,len(h)

def clean(t):
    return t.translate(str.maketrans('', '', '!"$%&\'()+/:;<=>?@[\\]^_`{|}~')).strip()
def cleant(t):
    return t.translate(str.maketrans('', '', string.punctuation)).strip()


def arrange(array):
    if len(array) == 1:
        return array
    ret = []
    intx = []
    for x in array:
        try:
            ret.append(intx.append(int(x)))
        except:
            ret.append(x)
    return ret
    
def extract3(text_detections_in_order,temp):    
    assert len(text_detections_in_order) <= 4
    for ind , x in enumerate(text_detections_in_order):
        if x != "":
            if ind == 0:
                poltotal = [int(word) for word in cleant(x).split() if word.isdigit()]
            if ind == 1:
                subprem = [int(word) for word in cleant(x).split() if word.isdigit()]
            if ind == 2:
                total_act_inc_loss = [int(word) for word in cleant(x).split() if word.isdigit()]
    try:
        polt = poltotal[0]
    except:
        polt = "0"
        
    try:
        subp = subprem[0]
    except:
        subp = "0"
        
    try:
        totala = total_act_inc_loss[0]
    except:
        totala = "0"

    temp["Policy Total"] = polt
    temp["Subject Premium"] = subp
    temp["Total Act Inc Losses"] = totala
    return temp
    
def extract4(LIST,data,r,c,b):
    temp = data.copy()
    CARRIER = [(c*0.07675,r-50+b),(c*0.14198,r+b)]
    POL = [(0.25864*c,r-50+b),(c*0.38373,r+b)]
    EFF = [(c*0.55488,r-50+b ),(c*0.65464,r+b)]
    EXP = [(c*0.801228,r-50+b),(0.90560*c,r+b)]
    STATECODE = [(0,r-100+b),(0.18*c,r-40+b)]
    
    ll = [CARRIER,POL,EFF,EXP,STATECODE]
    pol = ""
    carrier = ""
    eff = ""
    exp = ""
    state_code = ""
    STATEC = []
    EFFL = []
    EXPL = []
    
    
    for ind,x in enumerate(LIST):
        if blockIsIn(x[0],CARRIER):
            carrier = x[1]
        if blockIsIn(x[0],POL):
            pol = x[1]
        if blockIsIn(x[0],EFF):
            EFFL.append(x[1])
        if blockIsIn(x[0],EXP):
            EXPL.append(x[1])
        if blockIsIn(x[0],STATECODE):
            STATEC.append(x[1])
            
    state_code = "".join(STATEC)
    eff = "".join(EFFL)
    exp = "".join(EXPL)
    
    
    
    assert isinstance(carrier,str) and len(carrier) > 4
    assert isinstance(pol,str) and len(pol) > 4
    assert isinstance(eff,str) and len(eff) > 4
    assert isinstance(exp,str) and len(exp) > 4
    assert isinstance(state_code,str) and len(state_code) > 4

    temp["carrier"] = carrier
    temp["policy_no"] = pol
    temp["eff_date"] = eff
    temp["exp_date"] = exp
    temp["statecode"] = state_code
    return temp

def check1(text_detections_row):
    for x in text_detections_row:
        if "total" in x.lower().strip():
            return True
    return False

def hcheck(l):
    newl = [0]
    prev = newl[0]
    for x in range(1,len(l)):
        if abs(prev - l[x]) > 35:
            newl.append(l[x])
            prev = l[x]
    return newl
        
def secondpage(det):
    for (ptr,text) in det:
        if "stabiliz" in text.lower().strip():
            return True
    return False

path = sys.argv[-2]
json = {
            "id": "",
            "document_id": "",
            "remote_id": "",
            "file_name": path.split("/")[-1],
            "media_link": "",
            "media_link_original": "",
            "media_link_data": "",
            "page_count": "",
            "uploaded_at": "",
            "processed_at": "",
            "merged_amit_data_edit": []
}
im = convert_from_path(path)
json["page_count"] = str(len(im))

temp_list = []
polFound = True
for page in range(1,int(json["page_count"])):
    bound = boundary(im[page])
    bound = cv2.cvtColor(bound, cv2.COLOR_RGB2GRAY)
    bound = cv2.cvtColor(bound, cv2.COLOR_GRAY2RGB)
    
    r,c,_ = bound.shape
    
    if c > 2500:
        print("resizing")
        bound = cv2.resize(bound, (1300,1670), interpolation = cv2.INTER_AREA)
        
    r,c,_ = bound.shape
    if r < 1300:
        r = 1670
    
    secs,breakpoints = segmentation(bound,int(r*0.03588516746))
    if len(secs) > 1:
        small_block,cords,hlen = same(secs[-1],int(c*0.03846153846),int(c*0.02307692307))
        det = detect(bound,"page"+str(page),breakpoints,(cords[1][6][0][0]+5,cords[1][6][1][0]))
    else:
        det = detect(bound,"page"+str(page),breakpoints)
        
    
    
    if secondpage(det):
        continue
    sta = 1
    
    if not polFound and len(det) < 50:
        det = detect( bound [BreakPoints(bound,0,c-50)[1]: ,:] ,"page"+str(page),breakpoints)
        inorderminus1 = []
        for _,x in det:
            try:
                x = int(x.replace(",",""))
                inorderminus1.append("ABC "+str(x))
            except:
                pass
        inorderminus1.append("")
        inorderminus1.pop(0)
        temp = extract3(inorderminus1,t)
        for x in range(len(temp_list)-1,-1,-1):
            if temp_list[x]["Policy Total"] == 10101010101:
                temp_list[x]["Policy Total"] = temp["Policy Total"]
                temp_list[x]["Subject Premium"] = temp["Subject Premium"]
                temp_list[x]["Total Act Inc Losses"] = temp["Total Act Inc Losses"]
            else:
                break
        polFound = True
        continue

    elif not polFound:
        B = bound [BreakPoints(bound,0,c-20)[1]: ,:]
        secs,breakpoints = segmentation(B,int(r*0.03588516746))
        det = detect( B ,"page"+str(page),breakpoints)
        r,c,_ = B.shape
        
        if secs[0].shape[0] > 320:
            sta = 0
    else:
        t = extractsec1(det,c,breakpoints[1])
    print(len(secs),"secs in page",page)
    for sec_n in range(sta,len(secs)):
        sec = secs[sec_n].copy()
        small_block,cords,hlen = same(sec,int(c*0.03846153846),int(c*0.02307692307))
        print("Sec",sec_n,"of page",page,"has",hlen-1,"rows")
        text_detections_in_order = []
        for indr,row in enumerate(cords):
            text_detections_row = []
            for indc, ((tlr,tlc),(brr,brc)) in enumerate(row):
                text_detections_col = []
                for ind_d, (block_detection,text) in enumerate(det):
                    if blockIsIn(block_detection,((tlr,tlc+breakpoints[sec_n]),(brr,brc+breakpoints[sec_n]))):
                        text_detections_col.append(text)
                
                to_add = clean(" ".join(text_detections_col))
                if "com" in to_add.lower().strip():
                    continue
                text_detections_row.append(to_add)
            if check1(text_detections_row):
                text_detections_in_order.append(text_detections_row)
                break       
            if len(text_detections_row) != 1:
                text_detections_in_order.append(text_detections_row)
        text_detections_in_order[-1].pop(-1)
        if len(text_detections_in_order[-1]) < 5:
            temp = extract3(text_detections_in_order[-1],t)
            for x in range(len(temp_list)-1,-1,-1):
                if temp_list[x]["Policy Total"] == 10101010101:
                    temp_list[x]["Policy Total"] = temp["Policy Total"]
                    temp_list[x]["Subject Premium"] = temp["Subject Premium"]
                    temp_list[x]["Total Act Inc Losses"] = temp["Total Act Inc Losses"]
                else:
                    break
            polFound = True
        else:
            text_detections_in_order.append(["a 10101010101","a 10101010101","a 10101010101",""])
            temp = extract3(text_detections_in_order[-1],t)
            polFound = False

        for x in range(1,len(text_detections_in_order)-1):
            row = text_detections_in_order[x]
            if len(row) != 11:
                row.insert(2,"")
                row.insert(3,"")
                temp1 = extractsec3(row,temp)
            else:
                temp1 = extractsec3(row,temp)
            temp_list.append(temp1)
        try:
            t = extract4( det,t,sec.shape[0],c,breakpoints[sec_n]).copy()
        except:
            pass
                
print( "Total Records Found:", len(temp_list))
json["merged_amit_data_edit"] = temp_list
json_base = []
json_base.append(json)
with open(sys.argv[-1]+"/"+path.split("/")[-1].split(".")[0]+'.json', 'w') as fp:
    j.dump(json_base, fp,indent = 4)
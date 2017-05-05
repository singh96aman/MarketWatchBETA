from flask import Flask, session, redirect, url_for, escape, request, render_template, jsonify
import MySQLdb

app = Flask(__name__)

#######################
#   DATABASE CONFIG   #
#######################

db = MySQLdb.connect(host="localhost", user="root", passwd="24singh96", db="stockmarket")
cur = db.cursor()

@app.route('/')
def index():
    if 'username' in session:
        username_session = escape(session['username']).capitalize()
        return render_template('index.html', session_user_name=username_session)
    return redirect(url_for('login'))

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/signup', methods=['GET','POST'])
def SignUp():
    if request.method == 'POST':
        Signupuser=request.form['signupusername']
        Signuppass=request.form['signuppassword']
        temp = "Insert into login(name,password) values ('"+Signupuser+"','"+Signuppass+"');"
        cur.execute(temp)
        db.commit()
        return render_template('login.html')
    return render_template('signup.html')

@app.route('/changepass', methods=['GET','POST'])
def Changepass():
    if request.method == 'POST':
        Signupuser=request.form['signupusername']
        Signuppass=request.form['signuppassword']
        temp = "Update login set password='"+Signuppass+"' where name='"+Signupuser+"';"
        print temp
        cur.execute(temp)
        db.commit()
        return render_template('login.html')
    return render_template('change.html')

@app.route('/delete', methods=['GET','POST'])
def delete():
    if request.method == 'POST':
        Signupuser=request.form['signupusername']
        temp = "Delete from login where name='"+Signupuser+"';"
        print temp
        cur.execute(temp)
        db.commit()
        return render_template('login.html')
    return render_template("delete.html")


@app.route('/BSE', methods=['GET','POST'])
def bse():
    if request.method == 'POST':
        companyname = request.form['companyname']
        companyadd = request.form['addcompany']
        global list_companies
        list_companies=companyadd.split(",")
        if companyname != "":
            temp = "SELECT * from " + companyname
            cur.execute(temp)
            compdetails = [dict(
                dt=row[0],
                Open=row[1],
                High=row[2],
                Low=row[3],
                Close=row[4],
                Volume=row[5]
            ) for row in cur.fetchall()]
            return render_template('company.html', compdetails=compdetails)

    cur.execute('SELECT C_id, Cname, Sname FROM bse order by C_id ASC ')
    items = [dict(C_id=row[0],
                    Cname=row[1],
                    Sname=row[2]) for row in cur.fetchall()]
    return render_template('bse.html', items=items)

list_companies=""

@app.route('/NSE', methods=['GET','POST'])
def nse():
    if request.method == 'POST':
        companyname = request.form['companyname']
        companyadd = request.form['addcompany']
        global list_companies
        list_companies=companyadd.split(",")
        if companyname != "":
            temp = "SELECT * from " + companyname
            cur.execute(temp)
            compdetails = [dict(
                dt=row[0],
                Open=row[1],
                High=row[2],
                Low=row[3],
                Close=row[4],
                Volume=row[5]
            ) for row in cur.fetchall()]
            return render_template('company.html', compdetails=compdetails)

    cur.execute('SELECT C_id, Cname, Sname FROM NSE order by Cname')
    items2 = [dict(C_id=row[0],
                  Cname=row[1],
                  Sname=row[2]) for row in cur.fetchall()]
    return render_template('nse.html', items2=items2)

@app.route('/marketwatch')
def market():
    for i in list_companies:
        temp = "SELECT C_id, Cname, Sname, avg(Volume), avg(Close), avg(Volume)*avg(Close) as Capital from nse,"+i+" where C_id=\'"+i+"\'"
        cur.execute(temp)
        for row in cur.fetchall():
            marketwatch = [dict(
            C_id=row[0],
            Cname=row[1],
            Sname=row[2],
            Volume=row[3],
            Close=row[4],
            Capital=row[5]
            )]
        temp = "SELECT Close from "+i+""
        cur.execute(temp)
        for row in cur.fetchall():
            labels = [dict(
            Close=row[0]
            )]
    return render_template('watch.html',marketwatch=marketwatch,labels=labels)

@app.route('/marketwatch', methods=['GET','POST'])
def chut():
    if request.method == 'POST':
        comppredict = request.form['comppredict']
        return render_template('predict.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if 'username' in session:
        return redirect(url_for('index'))
    if request.method == 'POST':
        username_form  = request.form['username']
        password_form  = request.form['password']
        cur.execute("SELECT COUNT(1) FROM login WHERE name = %s;", [username_form]) # CHECKS IF USERNAME EXSIST
        if cur.fetchone()[0]:
            cur.execute("SELECT password FROM login WHERE name = %s;", [username_form])
            for row in cur.fetchall():
                if password_form == row[0]:
                    session['username'] = request.form['username']
                    return redirect(url_for('index'))
                else:
                    error = "Invalid Credential"
        else:
            error = "Invalid Credential"
    return render_template('login.html', error=error)

@app.route('/comppredict', methods=['GET','POST'])
def lala():
    for i in list_companies:
        Tp,list_out=SvmPredict(i)
    print Tp
    print list_out
    return render_template("predict.html",list_out=list_out,Tp=Tp)

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('index'))

app.secret_key = 'A0Zr98j/3yX R~XHH!jmN]LWX/,?RT'


def SvmPredict(filNam):
    import numpy as np
    from numpy import genfromtxt
    from sklearn import svm
    from sklearn.utils import shuffle
    import matplotlib.pyplot as plt
    from pandas.tools.plotting import scatter_matrix
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import StandardScaler
    op = []

    filepath = filNam + '.csv'
    path = "/Users/amansinghthakur/Flask/login/templates/csvfiles/" + filepath
    print path
    for z in range(10, 17):
        dataset = genfromtxt(path, delimiter=',')
        feat = dataset[:, :6]
        output = dataset[:, 7, None]
        feat, output = shuffle(feat, output, random_state=z)
        siz = dataset.shape[0]
        end = siz
        tr_end = siz / 2
        dele = np.zeros(0)
        for i in range(feat.shape[0]):
            if i < end:
                vat = np.isnan(feat)
                val = vat[i]
                if val[4] == True:
                    dele = np.append(dele, i)
        nfeat = feat
        j = 0
        for i in dele:
            nfeat = np.delete(nfeat, i - j, axis=0)
            j = j + 1

        # for i in range(nfeat.shape[0]):
        #   v = np.isnan(nfeat)
        #   if v[i,4] == True:
        #       print 'found'
        #   else:
        #       print 'not found'
        b = np.isinf(nfeat)
        dele = np.zeros(0)
        for i in range(nfeat.shape[0]):
            for j in range(nfeat.shape[1]):
                if b[i, j] == True:
                    dele = np.append(dele, i)
        j = 0
        for i in dele:
            nfeat = np.delete(nfeat, i - j, axis=0)
            j = j + 1

        standard_scaler = StandardScaler()
        feat = standard_scaler.fit_transform(nfeat)
        X_train = feat[10:tr_end, :]
        Y_train = output[10:tr_end, :]
        X_test = feat[1:10, :]
        Y_test = output[1:10, :]
        holdout = feat[tr_end:feat.shape[0], :]
        holdoutY = output[tr_end:output.shape[0], :]
        clf = svm.SVC(kernel='rbf', degree=9)
        clf.fit(X_train, Y_train)
        predicted = clf.predict(X_test)
        acc = accuracy_score(Y_test, predicted)
        # print acc
        op.append(acc)
    oput = clf.predict(feat[0, :].reshape((1, -1)))
    # print oput
    hand = 0
    for i in range(len(op)):
        if op[i] > hand:
            hand = op[i]
            best = op[i]
    out = [best, oput]
    Tp= feat[0,:].reshape((1,-1))
    return Tp,out

if __name__ == '__main__':
    app.run(port=5502)


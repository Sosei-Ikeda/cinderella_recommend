from django.shortcuts import render
import numpy as np
from collections import OrderedDict
    
def index(request):
    request.session['num_for'] = 0
    request.session['num_iter'] = 0
    return render(request, 'intelligence/index.html')

def calc(request):
    num_for = request.session.get('num_for') # 0<=num_for<=20
    num_iter = request.session.get('num_iter') #0<=num_iter<=10
    if num_for > 20:
        return render(request, '500.html', status=500)
    if num_for == 20:
        request.session['num_19'] = int(request.POST['num_post'])
        request.session['num_for'] = num_for+1
        request.session['num_iter'] = num_iter + 1
        addr_lis = ["nbydamrmiu",
                    "4gnd5ax2em",
                    "a4wx6iwmjt",
                    "6jamtxg969",
                    "ys7e8erhru",
                    "e363ysgbtr",
                    "iu9855px6s",
                    "jiz6jtu6m8",
                    "x493r73kfg",
                    "yrb2r9hbn6",
                    "7nhcfu9myc",
                    "7exer22bge",
                    "sephynxtgj",
                    "pnxnsip8p7",
                    "f9fct6ha8u",
                    "pkciixxm96",
                    "kacrby7uge",
                    "4gjbswztpw",
                    "6xy7zgkw2i",
                    "fm2bigfyat"]
        text_lis = ["夢見りあむお得意の早口リフレインですね。Pからアイドルにならないか誘われた際のセリフです。スカウトは困惑することだと思いますが全くの他人にこの反応…只者ではない…",
                    "コミュでPと初めて会ったときの自己紹介です。これが出発点の夢見りあむがすぐの総選挙で3位になったと考えると心中察するに余りあります。",
                    "夢見りあむ山盛りセットです。短い文章ですがこれでもかと夢見りあむのキャラクターが表現されています。",
                    "夢見りあむがドルオタなことは結構有名かと思いますが、その理由も本人の口から語られています。奇抜な設定ばかりが目を引きますが、実は自分に誇れるものを手に入れるという王道のストーリー展開です。",
                    "これは総選挙前のセリフです。大事なことなのでもう一度言いますがこれは総選挙前のセリフです。ここからあんなことになるとは誰も想像できなかったのではないでしょうか。",
                    "ネット民の注目を浴びて、”人気”ではないのに上位になる、珍しい現象ではないけれど他の投票モノと違って連続的な時間の進行にしたがって彼女には次がある。特有の誤解を生むヘタな表現ながら、彼女のこれからにぼんやりとした不安を抱かせる圧巻の受賞コメントです。",
                    "SSR実装です。総選挙について触れてきました。「ぼくのこと話題にして、ぼくじゃないぼくの話をしてた。」は当時衝撃を覚えました。ここでしっかり総選挙について触れたのは良くも悪くも夢見りあむにとっての転換点だったと思います。",
                    "SSRの特訓後です。ここでメンタルの強さを見せてくるのズルいと思いました。アイドルらしさに囚われるのをやめたりあむですが、このSSRのポスターでは「自分のためであってもステージを楽しむりあむの姿は誰よりアイドルらしい。りあむはまだ、それを知らない。」とあります。泣きました。",
                    "SSR+の親愛度MAX演出です。ここまでのりあむの歩みを考えると激エモです。あまり言葉はいらないと思うのでこのあたりで。",
                    "意外に思われるかもしれないと思ってこのセリフを選びました。アイドルらしくなくてもいいけどアイドルである以上理想を目指して努力することはやめない、というところがいいなあと思います。",
                    "他のアイドルとの絡みがほとんどない中、歌う一曲目が総選挙曲。プレッシャーの中で卑屈になりながらも他のアイドルたちに迎え入れられて仲間になる(Sun!High!Gold!)→仲間のアイドルたちと打ち解けて仲良くなる(夢をのぞいたら)と二つのコミュを通して丁寧に描かれてよかったです。",
                    "りあむ曰く「マウントとられまくり」の環境で育ったりあむがプロフィールの出身地の欄に書いたのは「やさしい世界」。本物の「やさしい世界」をりあむは見つけられたんだということが分かり微笑ましい瞬間です。",
                    "なんだこれ(褒め言葉)こういうところはずっと変わらないでいてほしいと思います。面白いので。",
                    "夢見りあむの面白さが出たいいコメントだと思います。優勝…優勝ってりあむお前…",
                    "宿屋のリポートの仕事で文香ちゃんにフォローしてもらった後にお礼を言いに行く場面です。こういう絡みをしてくるアイドルは珍しいと思うので他の担当Pさんも自担とりあむの絡み見たくないですか？どうですか？",
                    "「事務所の中にママを増やしていきたい…」と語っているのでりあむの登場が増えると他の担当Pさんも自担がママになる場面を見られるかもしれませんよ…？",
                    "夢をのぞいたらの歌詞を使いながらきれいに心境を表現しています。今はこの歌は自分には似合わないけれど「いつかこの歌を、ぼくの本当にしたいから…」という締めくくりはグッときます。",
                    "光をつかんだ未央、光に強気に手を伸ばす加蓮、控えめに手を伸ばすりあむ。それでもりあむが手を伸ばし始めた、ということに大きな意味があると思います。",
                    "「ふたり」は本田未央と北条加蓮のこと。「積み重ねてきたものも、誰かと歩いてきた道も、」まだないと語るりあむがこれから、彼女のことが「アイドルに見えた」Pたちと一緒にどのような道を歩んでいくのか期待が膨らむ一言。",
                    "これを見たとき震えましたね。デビューから1年、夢見りあむはここまで来ました。"]
        rand_num = np.random.randint(20)
        return render(request, 'intelligence/riamu.html', {'addr':addr_lis[rand_num], 'text':text_lis[rand_num]})
    else:
        lis = ["","","","","",""]
        if num_for != 0:
            num_post = int(request.POST['num_post'])
            request.session["num_{}".format(num_for-1)] = num_post
        if num_for%2 == 0:
            arange = np.arange(19)
            np.random.shuffle(arange)
            arange = 19*(num_iter)+arange
            var = []
            for i in range(6):
                var.append(int(arange[i]))
                request.session[f'var_{i}'] = int(var[i])
            text = "1番好きなのを選んでください"
        else:
            var = []
            for i in range(6):
                var.append(int(request.session.get(f'var_{i}')))
            for i in range(6):
                if var[i] == num_post:
                    lis[i] = "class = fav disabled"
                    break
            text = "6番目に好きなのを選んでください"
            request.session['num_iter'] = num_iter + 1
        request.session['num_for'] = num_for + 1
        context = {'var_0':var[0],'var_1':var[1],'var_2':var[2],'var_3':var[3],'var_4':var[4],'var_5':var[5],
                   'number':num_iter+1,'text':text,
                   'lis_0':lis[0],'lis_1':lis[1],'lis_2':lis[2],'lis_3':lis[3],'lis_4':lis[4],'lis_5':lis[5],
                   }
        return render(request, 'intelligence/calc.html',context)      

def result(request):
    def softmax(x):
        if x.ndim == 2:
            x = x.T
            x = x - np.max(x, axis=0)
            y = np.exp(x) / np.sum(np.exp(x), axis=0)
            return y.T 
    
        x = x - np.max(x)
        return np.exp(x) / np.sum(np.exp(x))
    
    def cross_entropy_error(y, t):
        if y.ndim == 1:
            t = t.reshape(1, t.size)
            y = y.reshape(1, y.size)
        t = t.argmax(axis=1)
    
        batch_size = y.shape[0]
    
        return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
    
    def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
        N, C, H, W = input_data.shape
        out_h = (H + 2*pad - filter_h)//stride + 1
        out_w = (W + 2*pad - filter_w)//stride + 1
    
        img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
        col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))
    
        for y in range(filter_h):
            y_max = y + stride*out_h
            for x in range(filter_w):
                x_max = x + stride*out_w
                col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
    
        col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
        return col
    
    def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
        N, C, H, W = input_shape
        out_h = (H + 2*pad - filter_h)//stride + 1
        out_w = (W + 2*pad - filter_w)//stride + 1
        col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)
    
        img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
        for y in range(filter_h):
            y_max = y + stride*out_h
            for x in range(filter_w):
                x_max = x + stride*out_w
                img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]
    
        return img[:, :, pad:H + pad, pad:W + pad]
    
    class Relu:
        def __init__(self):
            self.mask = None
    
        def forward(self, x):
            self.mask = (x <= 0)
            out = x.copy()
            out[self.mask] = 0
    
            return out
    
        def backward(self, dout):
            dout[self.mask] = 0
            dx = dout
    
            return dx
        
    class Affine:
        def __init__(self, W, b):
            self.W =W
            self.b = b
            
            self.x = None
            self.original_x_shape = None
            self.dW = None
            self.db = None
    
        def forward(self, x):
            self.original_x_shape = x.shape
            x = x.reshape(x.shape[0], -1)
            self.x = x
    
            out = np.dot(self.x, self.W) + self.b
    
            return out
    
        def backward(self, dout):
            dx = np.dot(dout, self.W.T)
            self.dW = np.dot(self.x.T, dout)
            self.db = np.sum(dout, axis=0)
            
            dx = dx.reshape(*self.original_x_shape)
            return dx  
    
    class SoftmaxWithLoss:
        def __init__(self):
            self.loss = None
            self.y = None
            self.t = None
    
        def forward(self, x, t):
            self.t = t
            self.y = softmax(x)
            self.loss = cross_entropy_error(self.y, self.t)
            
            return self.loss
    
        def backward(self, dout=1):
            batch_size = self.t.shape[0]
            dx = (self.y - self.t) / batch_size
            
            return dx
    
    class Convolution:
        def __init__(self, W, b, stride=1, pad=0):
            self.W = W
            self.b = b
            self.stride = stride
            self.pad = pad
            
            self.x = None   
            self.col = None
            self.col_W = None
            
            self.dW = None
            self.db = None
    
        def forward(self, x):
            FN, C, FH, FW = self.W.shape
            N, C, H, W = x.shape
            out_h = 1 + int((H + 2*self.pad - FH) / self.stride)
            out_w = 1 + int((W + 2*self.pad - FW) / self.stride)
    
            col = im2col(x, FH, FW, self.stride, self.pad)
            col_W = self.W.reshape(FN, -1).T
    
            out = np.dot(col, col_W) + self.b
            out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
    
            self.x = x
            self.col = col
            self.col_W = col_W
    
            return out
    
        def backward(self, dout):
            FN, C, FH, FW = self.W.shape
            dout = dout.transpose(0,2,3,1).reshape(-1, FN)
    
            self.db = np.sum(dout, axis=0)
            self.dW = np.dot(self.col.T, dout)
            self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)
    
            dcol = np.dot(dout, self.col_W.T)
            dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)
    
            return dx
    
    
    class Pooling:
        def __init__(self, pool_h, pool_w, stride=1, pad=0):
            self.pool_h = pool_h
            self.pool_w = pool_w
            self.stride = stride
            self.pad = pad
            
            self.x = None
            self.arg_max = None
    
        def forward(self, x):
            N, C, H, W = x.shape
            out_h = int(1 + (H - self.pool_h) / self.stride)
            out_w = int(1 + (W - self.pool_w) / self.stride)
    
            col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
            col = col.reshape(-1, self.pool_h*self.pool_w)
    
            arg_max = np.argmax(col, axis=1)
            out = np.max(col, axis=1)
            out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)
    
            self.x = x
            self.arg_max = arg_max
    
            return out
    
        def backward(self, dout):
            dout = dout.transpose(0, 2, 3, 1)
            
            pool_size = self.pool_h * self.pool_w
            dmax = np.zeros((dout.size, pool_size))
            dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
            dmax = dmax.reshape(dout.shape + (pool_size,)) 
            
            dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
            dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
            
            return dx
    
    class SimpleConvNet:
        def __init__(self, input_dim=(1, 28, 28), 
                     conv_param={'filter_num':30, 'filter_size':5, 'pad':0, 'stride':1},
                     hidden_size=100, output_size=10):
            filter_num = conv_param['filter_num']
            filter_size = conv_param['filter_size']
            filter_pad = conv_param['pad']
            filter_stride = conv_param['stride']
            input_size = input_dim[1]
            conv_output_size = (input_size - filter_size + 2*filter_pad) / filter_stride + 1
            pool_output_size = int(filter_num * (conv_output_size/2) * (conv_output_size/2))
    
            self.params = {}
            self.params['W1'] = np.sqrt(2.0/(input_dim[0]*filter_size*filter_size)) * \
                                np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
            self.params['b1'] = np.zeros(filter_num)
            self.params['W2'] = np.sqrt(2.0/(filter_num*filter_size*filter_size)) * \
                                np.random.randn(pool_output_size, hidden_size)
            self.params['b2'] = np.zeros(hidden_size)
            self.params['W3'] = np.sqrt(2.0/hidden_size) * \
                                np.random.randn(hidden_size, output_size)
            self.params['b3'] = np.zeros(output_size)
    
            self.layers = OrderedDict()
            self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'],
                                               conv_param['stride'], conv_param['pad'])
            self.layers['Relu1'] = Relu()
            self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
            self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2'])
            self.layers['Relu2'] = Relu()
            self.layers['Affine2'] = Affine(self.params['W3'], self.params['b3'])
    
            self.last_layer = SoftmaxWithLoss()
    
        def predict(self, x):
            for layer in self.layers.values():
                x = layer.forward(x)
    
            return x
    
        def loss(self, x, t):
            y = self.predict(x)
            return self.last_layer.forward(y, t)
    
        def gradient(self, x, t):
            self.loss(x, t)
    
            dout = 1
            dout = self.last_layer.backward(dout)
    
            layers = list(self.layers.values())
            layers.reverse()
            for layer in layers:
                dout = layer.backward(dout)
    
            grads = {}
            grads['W1'], grads['b1'] = self.layers['Conv1'].dW, self.layers['Conv1'].db
            grads['W2'], grads['b2'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
            grads['W3'], grads['b3'] = self.layers['Affine2'].dW, self.layers['Affine2'].db
    
            return grads
    
    class AdaGrad:
        def __init__(self, lr=0.01):
            self.lr = lr
            self.h = None
            
        def update(self, params, grads):
            if self.h is None:
                self.h = {}
                for key, val in params.items():
                    self.h[key] = np.zeros_like(val)
                
            for key in params.keys():
                self.h[key] += grads[key] * grads[key]
                params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)
    
    class Trainer:
        def __init__(self, network, x_train, t_train,
                     mini_batch_size=100,
                     optimizer_param={'lr':0.01}):
            self.network = network
            self.x_train = x_train
            self.t_train = t_train
            self.batch_size = mini_batch_size
            self.optimizer = AdaGrad(**optimizer_param)
            self.train_size = x_train.shape[0]
    
        def train_step(self):
            batch_mask = np.random.choice(self.train_size, self.batch_size)
            x_batch = self.x_train[batch_mask]
            t_batch = self.t_train[batch_mask]
    
            grads = self.network.gradient(x_batch, t_batch)
            self.optimizer.update(self.network.params, grads)
    
        def train(self):
            for i in range(20):
                self.train_step()

    
    num_list = []
    for i in range(20):
        num_list.append(int(request.session.get(f'num_{i}')))
    x_train = np.zeros((200,3,64,64))
    for i in range(20):
        x_train[i*10:(i+1)*10] = np.load(f"intelligence/data/{num_list[i]}.npy")
    t_train = np.load("intelligence/data/t.npy")
    
    network = SimpleConvNet(input_dim=(3,64,64), 
                            conv_param = {'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                            hidden_size=100, output_size=2)                     
    trainer = Trainer(network, x_train, t_train,
                      mini_batch_size=10,
                      optimizer_param={'lr': 0.01},)
    trainer.train()
    test = np.load("intelligence/data/ssr.npy")
    result = softmax(network.predict(test))
    result = np.argsort(result[:,0])
    
    ssr_array = ["［ステージオブマジック］島村卯月",
                 "［ラブ！ストレート］中野有香",
                 "［出会えた憧憬］関裕美",
                 "［笑うたい樹］上田鈴帆",
                 "［イン・ヒューマン］高峯のあ",
                 "［超☆志貫徹］冴島清美",
                 "［P.C.S］五十嵐響子",
                 "［すいすいオーシャン］浅利七海",
                 "［シーサイドドリーム］荒木比奈",
                 "［お祭り小町大作戦］桃井あずき",
                 "［夢うたうチカラ］南条光",
                 "［P.C.S］島村卯月",
                 "［お菓子なドリーミング］喜多日菜子",
                 "［わきわきわんだほー］棟方愛海",
                 "［なよ竹の美器］藤原肇",
                 "［巡り会えた彩り］鷹富士茄子",
                 "［ライドオンステージ］原田美世",
                 "［フリースタイル☆ライフ］小松伊吹",
                 "［individuals］星輝子",
                 "［羽衣小町］小早川紗枝",
                 "［individuals］森久保乃々",
                 "［individuals］早坂美玲",
                 "［ウォーミング・ハート］アナスタシア",
                 "［ディーラーズディライト］兵藤レナ",
                 "［ラブモリ☆パーリィー］藤本里奈",
                 "［羽衣小町］塩見周子",
                 "［ステップ・マイ・ステップ］福山舞",
                 "［ポジティブパッション］本田未央",
                 "［エレガンス・プラス］十時愛梨",
                 "［イノセント・ホワイト］櫻井桃華",
                 "［色とりどりのゆめ］成宮由愛",
                 "［舞台上のエレガンテ］松山久美子",
                 "［しゅまいるプレゼント］大沼くるみ",
                 "［Triad Primus］渋谷凛",
                 "［にこにこキッチン］龍崎薫",
                 "［ようせいのこ］遊佐こずえ",
                 "［音色を抱きしめて］水本ゆかり",
                 "［無限大オーバーフロー］真鍋いつき",
                 "［ブランニュー・ビート］多田李衣菜",
                 "［緋薔薇の令嬢］黒埼ちとせ",
                 "［白面の騎士］白雪千夜",
                 "［ノブレス・ノワール］黒川千秋",
                 "［星へのピチカート］涼宮星花",
                 "［オフタイム・ナギルーム］久川凪",
                 "［オンタイム・ハーモニー］久川颯",
                 "［歓びのアティチュード］綾瀬穂乃香",
                 "［ミスティックエリクシル］一ノ瀬志希",
                 "［いっぱいの感謝］相原雪乃",
                 "［エール・フォー・オール］若林智香",
                 "［わたし色の羽根で］白菊ほたる",
                 "［アップビート・ラン］北川真尋",
                 "［ほわあまプリンセス］榊原里美",
                 "［アドバンス・アバンチュール］川島瑞樹",
                 "［夜更けに咲く才媛］和久井留美",
                 "［おめかし☆あかりんご］辻野あかり",
                 "［エキゾチックプレイヤー］杉坂海",
                 "［想いは深く、歌声は遠く］瀬名詩織",
                 "［キャットパーティー］前川みく",
                 "［カントリー・ステップ］関裕美",
                 "［レットイットキュート！］メアリー・コクラン",
                 "［ライフ・オンライン］砂塚あきら",
                 "［ユリユリフェスティバル］大西由里子",
                 "［プールサイド・マーメイド］西島櫂",
                 "［トップ・オブ・ウェーブ］沢田麻理菜",
                 "［夢見りあむは救われたい］夢見りあむ",
                 "［幸せのお告げ］藤居朋",
                 "［風のゆくさき］浅野風香",
                 "［重ねた手と日々］佐城雪美",
                 "［アベニュー・モード］宮本フレデリカ",
                 "［ブラッシング・ユー！］太田優",
                 "［学び舎のペルソナ］八神マキノ",
                 "［笑いの中心］キャシー・グラハム",
                 "［みんなの人気者］赤西瑛梨華",
                 "［戦国愛☆一代］丹羽仁美",
                 "［喝采のディーヴァ］西川保奈美",
                 "［華美爛漫］楊菲菲",
                 "［モード・ラフィネ］相川千夏",
                 "［ライジングビート］仙崎恵磨",
                 "［ライフイズアート］吉岡沙紀",
                 "［花舞うまほろば］小早川紗枝",
                 "［こずえからみんなへ］遊佐こずえ",
                 "［トワ・エ・モワ］柊志乃",
                 "［アットホーム・マイルーツ］ナターリア",
                 "［私に恋をする私］井村雪菜",
                 "［聖歌の天使］望月聖",
                 "［影に揺れる光］服部瞳子",
                 "［ごーごー！すまいらー！！］野々村そら",
                 "［みんなのお姉さん］持田亜里沙",
                 "［あの日の私を受け止めて］岡崎泰葉",
                 "［スペシャルセクシー］松本沙理奈",
                 "［ピュア・ユーフォリア］西園寺琴歌",
                 "［見つめて★レディ］的場梨沙",
                 "［あるがまま咲く花］江上椿",
                 "［マックスビューティー］斉藤洋子",
                 "［早耶の瞳に恋してる］松原早耶",
                 "［みゆきとあそぼ！］柳瀬美由紀",
                 "［華麗なパピヨン］岸部彩華",
                 "［トゥインクル☆ラブリー］横山千佳",
                 "［愛・ビリーブ］有浦柑奈",
                 "［あたしコーディネート］衛藤美紗希",
                 "［憧れのメドレー］長富蓮実",
                 "［ぐうたら王国］双葉杏",
                 "［この身ひとつで］木場真奈美",
                 "［きらめきの幕開け］松尾千鶴",
                 "［エアリアルメロディア］水本ゆかり",
                 "［自称・カンペキ］輿水幸子",
                 "［夢追い人の光］工藤忍",
                 "［あこがれのプリンセス］古賀小春",
                 "［祝福のまなざし］クラリス",
                 "［フィールマイハート］佐久間まゆ",
                 "［手折られぬ花］白菊ほたる",
                 "［メイク★インパクト］早坂美玲",
                 "［スターティングデイズ］乙倉悠貴",
                 "［ステージオブマジック］渋谷凛",
                 "［アンビバレント・アクト］川島瑞樹",
                 "［Ring♪Ring♪フィーリン］椎名法子",
                 "［オーバー・ザ・レインボー］神谷奈緒",
                 "［澄みきった世界］上条春菜",
                 "［ネクスト☆ページ］荒木比奈",
                 "［目をあけてみる夢］多田李衣菜",
                 "［ステップトゥーミライ］佐々木千枝",
                 "［ルージュクチュール］三船美優",
                 "［ただひとつの器］藤原肇",
                 "［ノーブルヴィーナス］新田美波",
                 "［ありすのティーパーティー］橘ありす",
                 "［ブライトメモリーズ］鷺沢文香",
                 "［メモリアルデイズ］今井加奈",
                 "［インサイト・エクステンド］八神マキノ",
                 "［異国に吹く風］ライラ",
                 "［SOUND A ROUND］松永涼",
                 "［等身大の距離で］高垣楓",
                 "［薔薇の闇姫］神崎蘭子",
                 "［煌めきのひととき］北条加蓮",
                 "［ひみつの小夜曲］佐城雪美",
                 "［センリツノヨル］白坂小梅",
                 "［夜色の暁風］塩見周子",
                 "［真剣・一閃］脇山珠美",
                 "［ドルチェ・クラシカ］三村かな子",
                 "［エンドレスナイト］速水奏",
                 "［森のものがたり］森久保乃々",
                 "［星巡る物語］アナスタシア",
                 "［コマンドー・オブ・ステージ］大和亜季",
                 "［《偶像》のフラグメント］二宮飛鳥",
                 "［オーダー・フォー・トップ］桐生つかさ",
                 "［ステージオブマジック］本田未央",
                 "［てづくりのしあわせ］高森藍子",
                 "［ひまわりサニーデイ］龍崎薫",
                 "［FEEL SO FREE］木村夏樹",
                 "［P.C.S］小日向美穂",
                 "［メイクミー・キスユー］赤城みりあ",
                 "［ロリポップ・ハニー］大槻唯",
                 "［フルスイング☆エール］姫川友紀",
                 "［ハイテンションスマッシュ］喜多見柚",
                 "［はつらつハーヴェスト］及川雫",
                 "［マッシュアップ★ボルテージ］星輝子",
                 "［セクシーBANG☆BANG］片桐早苗",
                 "［セーシュンエナジー］堀裕子",
                 "［ビビッド★エゴイスト］的場梨沙",
                 "［クイーン・オブ・クイーン］財前時子",
                 "［マイ・フェアリーテイル］緒方智絵里",
                 "［わだつみの導き手］依田芳乃",
                 "［束ねた気持ち］相葉夕美",
                 "［キラデコ☆パレード］城ヶ崎莉嘉",
                 "［ポジティブパッション］日野茜",
                 "［グレイトプレゼント］諸星きらり",
                 "［カップオブラブ］十時愛梨",
                 "［ヒートビート・サンバ！］ナターリア",
                 "［桃色怒髪天］向井拓海",
                 "［ともだちたくさん］市原仁奈",
                 "［夢みるプリンセス］喜多日菜子",
                 "［あったかハート］五十嵐響子",
                 "［プレイ・ザ・ゲーム！］三好紗南",
                 "［本日の主役］難波笑美",
                 "［くのいちのいろは］浜口あやめ",
                 "［こぶし・紅］村上巴",
                 "［はぁとトゥハート］佐藤心",
                 "［ドリーミン☆ウサミン］安部菜々",
                 "［心躍るアドベンチャー］氏家むつみ",
                 "［ニューウェーブ・ネイビー］大石泉",
                 "［センター・オブ・ストリート］城ヶ崎美嘉",
                 "［すこやかな願い］栗原ネネ",
                 "［ディアマイレディ］櫻井桃華",
                 "［ニューウェーブ・ピーチ］村松さくら",
                 "［バーン・アンド・ドーン！！］小関麗奈",
                 "［ブレイク・ワン・モーメント］水木聖來",
                 "［ニューウェーブ・サルファー］土屋亜子",
                 "［プレイ・ウィズ・ミー］結城晴",
                 "［ひたむきな歩み］道明寺歌鈴",
                 "［ビター・エレガンス］東郷あい",
                 "［パンパンの夢］大原みちる",
                 "［ときめきトラベラー］並木芽衣子",
                 "［ひらめきロボティクス］池袋晶葉",]

    context = {
        '1st': result[191],
        '2nd': result[190],
        '3rd': result[189],
        '4th': result[188],
        '5th': result[187],
        '1st_name': ssr_array[result[191]],
        '2nd_name': ssr_array[result[190]],
        '3rd_name': ssr_array[result[189]],
        '4th_name': ssr_array[result[188]],
        '5th_name': ssr_array[result[187]],
    }
    return render(request, 'intelligence/result.html', context)
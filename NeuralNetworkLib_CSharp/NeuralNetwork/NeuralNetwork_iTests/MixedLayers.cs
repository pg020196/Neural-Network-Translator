using System;
using System.Collections.Generic;
using NeuralNetwork;
using Layers;
using Tensor;

namespace NeuralNetwork //_iTests
{
	public class MixedLayers
    {
        private readonly int myNumLayers;
        private readonly int[] myLayerOutputHeight;
        private readonly int[] myLayerOutputWidth;
        private readonly int[] myLayerOutputDepth;
        private readonly int[] myLayerTypes;
        private readonly int[] myActivationTypes;
        private readonly Tensor<double> myWeights;
        private readonly int[] myIndicesWeights;
        private readonly Tensor<double> myBias;
        private readonly int[] myIndicesBias;
        private readonly int[] myUseBias;
        private readonly int[] myPoolHeights;
        private readonly int[] myPoolWidths;
        private readonly int[] myVerticalStride;
        private readonly int[] myHorizontalStride;
        private readonly int[] myPadding;

        public NeuralNetwork<double> NeuralNetwork { get; }

        static private ActivationType DecodeActivationType(int code)
        {
            switch (code)
            {
                case (0):
                    return ActivationType.linear;
                case (1):
                    return ActivationType.sigmoid;
                case (2):
                    return ActivationType.relu;
                case (3):
                    return ActivationType.tanh;
                case (4):
                    return ActivationType.softmax;
                default:
                    throw new NotImplementedException();
            }
        }

        public MixedLayers()
        {
            myNumLayers = 4;
            myLayerOutputHeight = new int[] {0,0,16,10};
            myLayerOutputWidth = new int[] {0,0,0,0};
            myLayerOutputDepth = new int[] {0,0,0,0};
            myLayerTypes = new int[] {4,2,1,1};
            myActivationTypes = new int[] {0,0,2,4};
            myWeights = new Tensor<double>(new double[] {0.38721179962158203,0.295003205537796,0.14998704195022583,-1.009785532951355,-0.1655689924955368,-0.013597238808870316,0.6954039335250854,-0.29370594024658203,0.08169697970151901,-0.9748908281326294,-1.2653913497924805,-1.104623556137085,-0.36494511365890503,0.3592069745063782,-0.2588666081428528,0.040996648371219635,0.24828670918941498,0.5173084139823914,-0.14650583267211914,-0.5759146213531494,0.061054158955812454,0.22036567330360413,1.0184756517410278,0.31780102849006653,0.9555561542510986,0.8040205240249634,-0.7482494115829468,-0.40499913692474365,0.5357264280319214,0.7337609529495239,0.1319543868303299,-0.5825002789497375,0.4927915632724762,0.323676735162735,0.33386242389678955,-0.7127900123596191,-0.3534889221191406,-0.687896728515625,1.1010297536849976,0.18620173633098602,-0.5302842855453491,0.2716858386993408,-0.5552998781204224,0.658880352973938,-0.004659928381443024,0.15210987627506256,-0.4836133122444153,-0.3440641462802887,0.8522108793258667,0.35227036476135254,0.0035724614281207323,-0.2565843462944031,-0.13804051280021667,-0.6847394704818726,0.6559723019599915,-0.7407916784286499,0.08292670547962189,-0.01113047730177641,-0.8291842341423035,-0.5725206136703491,0.28297939896583557,-0.8489173054695129,0.21473383903503418,-0.16475935280323029,-0.5836287140846252,0.30552148818969727,0.2987712323665619,0.1703311949968338,-0.733703076839447,-0.030539298430085182,-0.952642023563385,0.137959286570549,-0.28061240911483765,-0.8041345477104187,0.1846507489681244,0.3766603171825409,0.049818798899650574,-0.014339471235871315,0.3361052870750427,-0.6738654971122742,-0.12613455951213837,0.48411181569099426,1.0237958431243896,0.9534108638763428,-0.5177182555198669,0.05904434993863106,-1.2352633476257324,-0.33642932772636414,0.4418871998786926,-0.23913182318210602,1.1021068096160889,-0.12457390129566193,1.0294532775878906,-0.7983028888702393,0.3194650411605835,-0.4145202934741974,0.33372730016708374,-1.3786166906356812,0.9342830777168274,-0.29586663842201233,-0.15563364326953888,0.42200812697410583,-0.430578351020813,-1.2247240543365479,-0.2840869128704071,0.16744908690452576,-0.3612830936908722,0.9406750202178955,-0.15546433627605438,0.7873913049697876,0.7928341627120972,0.009328525513410568,-0.8714207410812378,-0.5443398356437683,-0.14731715619564056,0.3547075390815735,-0.29162710905075073,0.9786953926086426,-0.4986039102077484,-0.6706250309944153,-1.0712324380874634,0.08733242005109787,-1.0776479244232178,0.5415247082710266,0.06160128116607666,0.8049638271331787,0.32994386553764343,0.8526267409324646,-1.126395344734192,0.3151484429836273,-1.467240333557129,-0.21159109473228455,0.06610622256994247,1.2161535024642944,-0.5204653739929199,0.5522238612174988,-0.42262691259384155,0.37072843313217163,0.1844991147518158,-0.5841999053955078,-0.2149410992860794,-0.37079355120658875,0.23670724034309387,0.7540795803070068,-0.8488098382949829,-0.1534387171268463,0.2224642038345337,1.0944708585739136,1.0821670293807983,-0.4107041656970978,-0.6172699928283691,-0.5564084649085999,-0.30575454235076904,1.0305638313293457,-0.2324976921081543,0.11172524094581604,-0.6233656406402588,-0.12650556862354279,0.15191619098186493,0.40157267451286316,0.717793345451355,-0.18853642046451569,0.3232296109199524,1.0578073263168335,-0.7640182375907898,-0.3541189134120941,0.17269138991832733,0.31521594524383545,-0.6599335670471191,-0.5982125401496887,0.77431321144104,-0.7182930707931519,-0.7787180542945862,-0.7556191682815552,0.5633162260055542,0.5612921714782715,1.1544657945632935,0.4321095049381256,-1.020289659500122,0.47717174887657166,-0.2510942816734314,1.384566307067871,0.10145453363656998,0.4621264636516571,-0.48459547758102417,-0.5022271275520325,-0.21043258905410767,-0.29398661851882935,0.7346379160881042,-0.6549081802368164,0.1618371605873108,0.019154392182826996,-0.018031589686870575,0.17234711349010468,0.3095230460166931,-0.2655903995037079,-0.2581397294998169,-0.42540425062179565,0.12115774303674698,-0.22011041641235352,0.3241538107395172,-0.5845820903778076,-0.7621681690216064,-0.6141539216041565,-0.04802572727203369,0.33344313502311707,-0.7459055781364441,0.9441788196563721,-0.3160254657268524,0.29961374402046204,-0.6071827411651611,0.6900056600570679,1.1213366985321045,0.28555047512054443,-0.023955749347805977,0.4415384531021118,1.5138084888458252,0.31087231636047363,-0.4261046350002289,0.15366485714912415,-0.09777721017599106,0.5537112951278687,-0.6408590078353882,0.3114580810070038,-0.20205673575401306,-0.24251654744148254,-0.5852869153022766,0.3313446640968323,0.29480066895484924,-0.11449158936738968,-1.0393742322921753,1.1024835109710693,0.2655632793903351,-0.38402271270751953,0.10376427322626114,0.43070724606513977,-0.5012334585189819,0.7484904527664185,-1.7751442193984985,-1.1350868940353394,-0.2324623167514801,-0.9567484259605408,-0.8771654963493347,0.20218470692634583,0.2536959648132324,-0.5118063688278198,0.19925850629806519,-0.7612755298614502,1.122127890586853,-0.7194393873214722,0.2159794420003891,-1.287774682044983,-1.1886248588562012,0.7116752862930298,-0.20563364028930664,-0.5352947115898132,-2.200037717819214,0.11893032491207123,-0.8585311770439148,1.1994130611419678,-1.0717722177505493,0.1714722365140915,-0.4098357558250427,0.1278190314769745,0.09503179788589478,0.5127292275428772,0.12372943758964539,-0.3887862265110016,-0.67794269323349,-0.37889543175697327,-0.32848018407821655,1.2753735780715942,0.12069602310657501,-0.3154429495334625,0.06366611272096634,-1.4628483057022095,-0.6330162286758423,0.587539792060852,-1.453019142150879,-0.7182071805000305,0.10096889734268188,-0.10703347623348236,-1.136657953262329,-0.5558781623840332,0.9327888488769531,0.23236899077892303,0.16629920899868011,-0.022906985133886337,-1.3769196271896362,-0.908470094203949,0.753971517086029,-0.4994190037250519,1.3114532232284546,-0.7487491369247437,0.17686232924461365,-0.48824042081832886,-0.4955350458621979,0.36387255787849426,0.943442702293396,-0.43759769201278687,-1.217813491821289,-1.1084667444229126,1.1851340532302856,-0.3137427866458893,0.029470419511198997,-1.45442533493042,1.0391181707382202,-1.8614418506622314,-1.4432216882705688,-0.1416713446378708,0.8536362051963806,0.09811722487211227,0.07950231432914734,-0.1860181838274002,-0.9408362507820129,0.5426174998283386,-0.6655306220054626,-1.0847245454788208,0.3377223610877991,0.8602277040481567,-1.2777882814407349,-0.42353737354278564,-1.390676498413086,0.7810101509094238,-0.7183147072792053,-0.39183297753334045,1.0634485483169556,-0.6966544389724731,0.17422838509082794,1.0884805917739868,0.13437750935554504,0.3523869812488556,0.06139739975333214,-1.2478245496749878,-0.6024372577667236,-1.0515389442443848,-1.3082789182662964,0.7192808389663696,0.4652278423309326,-0.5085113644599915,-1.0363743305206299,0.388629287481308,-0.5066744089126587,-1.1856902837753296,-0.26149579882621765,0.032265160232782364,0.17376603186130524,0.4006808400154114,-0.35493138432502747,-0.23927605152130127,-0.9302317500114441,-0.4379889965057373,-0.42040154337882996,0.17272579669952393,0.270579993724823,-1.6973230838775635,-0.612812340259552,0.8200247883796692,-1.4048494100570679,-0.03202373534440994,0.5645628571510315,0.09263245761394501,0.4431285560131073,-0.8908799886703491,-0.14425334334373474,-0.37629446387290955,0.4686480462551117,0.8118293881416321,0.7864084243774414,-0.6203425526618958,-0.7924594879150391,-1.1980990171432495,-0.8739073872566223,-0.4544101357460022,-0.3544584810733795,0.4836241602897644,0.7801001071929932,-0.5655835270881653,-0.918710470199585,-0.8904158473014832,-0.6813903450965881,0.8773764371871948,-0.31260946393013,-0.391791969537735,-0.28132110834121704,0.6723299622535706,0.22781319916248322,-0.6205769777297974,1.107033371925354,0.2318083494901657,-1.3539707660675049,-0.8437255620956421,-1.7949200868606567,-0.13345089554786682,0.4865207374095917,1.0104490518569946,-0.6465832591056824,-0.4066445231437683,-0.958232581615448,-1.2914533615112305,0.9964927434921265,-0.7107405662536621,-0.8633826375007629,1.1743528842926025,-0.9759814739227295,0.5552517771720886,-0.7516189813613892,-0.17946843802928925,0.1832411289215088,-0.27077940106391907,1.2871227264404297,-0.8504131436347961,-1.0425928831100464,0.9768381714820862,-0.08175650238990784,-0.7443712949752808});
            myIndicesWeights = new int[] {0,0,0,256};
            myBias = new Tensor<double>(new double[] {0.9289207458496094,0.7958607077598572,0.9329409599304199,0.9043174982070923,0.7117593288421631,0.7914996147155762,0.5333093404769897,0.6894113421440125,0.24119248986244202,0.557752788066864,0.20272782444953918,0.25170716643333435,0.7237936854362488,1.2452476024627686,0.1028619036078453,0.0899728387594223,0.04624581336975098,-0.5790086388587952,0.18906021118164062,0.27005505561828613,-0.22784599661827087,0.0025378703139722347,-0.12324671447277069,-0.650705873966217,0.7530146837234497,-0.13174036145210266});
            myIndicesBias = new int[] {0,0,0,16};
            myUseBias = new int[] {0,0,1,1};
            myPoolHeights = new int[] {2,0,0,0};
            myPoolWidths = new int[] {2,0,0,0};
            myVerticalStride = new int[] {2,0,0,0};
            myHorizontalStride = new int[] {2};
            myPadding = new int[] {0,0,0,0};

            int inputHeight = myLayerOutputHeight[0];
            int inputWidth = myLayerOutputWidth[0];
            int inputDepth = myLayerOutputDepth[0];

            int[] inputShape;
            if (inputWidth == 0)
            {
                 inputShape = new int[] { inputHeight };
            }
            else if (inputDepth == 0)
            {
                inputShape = new int[] { inputHeight, inputWidth };
            }
            else
            {
                inputShape = new int[] { inputHeight, inputWidth, inputDepth };
            }

            var inputLayer = new InputLayer<double>(inputShape);

            var layerList = new List<BaseLayer<double>>();
            layerList.Add(inputLayer);

            BaseLayer<double> previousLayer = inputLayer;

            for (int layerIndex = 1; layerIndex <= myNumLayers; layerIndex++)
            {
                switch (myLayerTypes[layerIndex - 1])
                {
                    case (1):
                        var denseLayer = CreateDenseLayer(layerIndex, previousLayer);
                        layerList.Add(denseLayer);
                        previousLayer = denseLayer;
                        break;
                    case (2): // flatten
                        var flattenLayer = new Flatten<double>(previousLayer.OutputShape);
                        layerList.Add(flattenLayer);
                        previousLayer = flattenLayer;
                        break;
                    case (3): // maxpooling
                        var maxPoolingLayer = CreateMaxPoolingLayer(layerIndex, previousLayer);
                        layerList.Add(maxPoolingLayer);
                        previousLayer = maxPoolingLayer;
                        break;
                    case (4): // avgpooling
                        var averagePoolingLayer = CreateAveragePoolingLayer(layerIndex, previousLayer);
                        layerList.Add(averagePoolingLayer);
                        previousLayer = averagePoolingLayer;
                        break;
                    default:
                        throw new NotImplementedException($"Layer type {myLayerTypes[layerIndex - 1]} not implemented");
                }
            }

            NeuralNetwork = new NeuralNetwork<double>(layerList);
        }

        private Dense<double> CreateDenseLayer(int layerIndex, BaseLayer<double> prevLayer)
        {
            int weightsIndexUpper;
            int biasIndexUpper;
            if (layerIndex == myNumLayers)
            {
                weightsIndexUpper = myWeights.Shape[0];
                biasIndexUpper = myBias.Shape[0];
            }
            else
            {
                weightsIndexUpper = myIndicesWeights[layerIndex];
                biasIndexUpper = myIndicesBias[layerIndex];
            }

            var activationType = DecodeActivationType(myActivationTypes[layerIndex - 1]);
            var useBias = Convert.ToBoolean(myUseBias[layerIndex - 1]);
            var denseLayer = new Dense<double>(prevLayer.OutputShape, myLayerOutputHeight[layerIndex-1], activationType, useBias);

            int[] weightsShape = { prevLayer.OutputShape[0], myLayerOutputHeight[layerIndex-1] };
            var layerWeights = myWeights[myIndicesWeights[layerIndex - 1]..weightsIndexUpper];
            layerWeights = layerWeights.reshape(weightsShape);

            int biasShape = myLayerOutputHeight[layerIndex-1];
            var layerBias = myBias[myIndicesBias[layerIndex - 1]..biasIndexUpper];
            layerBias = layerBias.reshape(biasShape);

            denseLayer.Weights = layerWeights;
            denseLayer.Bias = layerBias;

            return denseLayer;
        }

        private BaseLayer<double> CreateMaxPoolingLayer(int layerIndex, BaseLayer<double> previousLayer)
        {
            var poolHeight = myPoolHeights[layerIndex - 1];
            var poolWidth = myPoolWidths[layerIndex - 1];
            var is1dPooling = poolHeight != 0 && poolWidth == 0;
            var is2dPooling = poolHeight != 0 && poolWidth != 0;

            PaddingType paddingType;
            if (myPadding[layerIndex - 1] == 0)
            {
                paddingType = PaddingType.valid;
            }
            else
            {
                paddingType = PaddingType.same_keras;
            }

            if (is1dPooling)
            {
                int stride = myVerticalStride[layerIndex - 1];
                var maxPoolingLayer = new PoolingLayer1D<double>(previousLayer.OutputShape, PoolingType.max, poolHeight, stride, paddingType);
                return maxPoolingLayer;
            }
            else if (is2dPooling)
            {
                int[] stride = { myVerticalStride[layerIndex - 1], myHorizontalStride[layerIndex - 1] };
                int[] poolSize = { poolHeight, poolWidth };
                var maxPoolingLayer = new PoolingLayer2D<double>(previousLayer.OutputShape, PoolingType.max, poolSize, stride, paddingType);
                return maxPoolingLayer;
            }
            else
            {
                throw new NotSupportedException($"Layer {layerIndex} is average pooling, but pool height and width are zero.");
            }
        }

        private BaseLayer<double> CreateAveragePoolingLayer(int layerIndex, BaseLayer<double> previousLayer)
        {
            var poolHeight = myPoolHeights[layerIndex - 1];
            var poolWidth = myPoolWidths[layerIndex - 1];
            var is1dPooling = poolHeight != 0 && poolWidth == 0;
            var is2dPooling = poolHeight != 0 && poolWidth != 0;

            PaddingType paddingType;
            if (myPadding[layerIndex - 1] == 0)
            {
                paddingType = PaddingType.valid;
            }
            else
            {
                paddingType = PaddingType.same_keras;
            }

            if (is1dPooling)
            {
                int stride = myVerticalStride[layerIndex - 1];
                var averagePoolinglayer = new PoolingLayer1D<double>(previousLayer.OutputShape, PoolingType.average, poolHeight, stride, paddingType);
                return averagePoolinglayer;
            }
            else if (is2dPooling)
            {
                int[] stride = { myVerticalStride[layerIndex - 1], myHorizontalStride[layerIndex - 1] };
                int[] poolSize = { poolHeight, poolWidth };
                var averagePoolinglayer = new PoolingLayer2D<double>(previousLayer.OutputShape, PoolingType.average, poolSize, stride, paddingType);
                return averagePoolinglayer;
            }
            else
            {
                throw new NotSupportedException($"Layer {layerIndex} is average pooling, but pool height and width are zero.");
            }
        }
    }
}
function drawChartLoss() {
    console.log(historyData,"----")
    var data = new google.visualization.DataTable();
    data.addColumn('number', 'Day');
    data.addColumn('number', 'Train Loss');
    data.addColumn('number', 'Validation Loss');

    var loss = historyData.loss
    var val_loss = historyData.val_loss

    for (let index = 0; index < loss.length; index++) {
        console.log(index)
        var temp = ([(index+1),loss[index],val_loss[index] ])
        data.addRow(temp)
        
    }

    var options = {
      chart: {
        title: '- - Loss Chart - -',
        subtitle: 'Train and Validation Sets'
      },
      width: 800,
      height: 400,
      backgroundColor: '#f1f1f1',
      chartArea: {
        backgroundColor: {
            'fill': '#f1f1f1',
            'opacity': 100
         },
     }
    };

    var chart = new google.charts.Line(document.getElementById('loss_chart'));

    chart.draw(data, google.charts.Line.convertOptions(options));
  }

  // ------------- ACCURACY -------------- //
  function drawChartAccuracy() {

    var data = new google.visualization.DataTable();
    data.addColumn('number', 'Day');
    data.addColumn('number', 'Train Accuracy');
    data.addColumn('number', 'Validation Accuracy');


    var accuracy = historyData.accuracy
    var val_accuracy = historyData.val_accuracy

    for (let index = 0; index < accuracy.length; index++) {
        var temp = ([(index+1),accuracy[index],val_accuracy[index] ])
        data.addRow(temp)
        
    }

    var options = {
        chart: {
          title: '- - Accuracy Chart - -',
          subtitle: 'Train and Validation Sets'
        },
        width: 800,
        height: 400,
        backgroundColor: '#f1f1f1',
        chartArea: {
          backgroundColor: {
              'fill': '#f1f1f1',
              'opacity': 100
           },
       }
      };

    var chart = new google.charts.Line(document.getElementById('accuracy_chart'));

    chart.draw(data, google.charts.Line.convertOptions(options));
  }
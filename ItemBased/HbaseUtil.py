# -*- coding: UTF-8 -*-
# add by Kylin 
def flatRow(row,key_col,column_family,column_names):
    rowkey = str(row[key_col])
    result = [] 
    for col in column_names:
        one_col = [] 
        if col == key_col:
            pass
        else:
            if row[col] is None :
                pass 
            else :
                one_col.append(rowkey)
                one_col.append(column_family)
                one_col.append(col)
                one_col.append(str(row[col]))
        if len(one_col) > 0 : 
            result.append((rowkey,one_col))
    return result

def write2hbase(df,hbase_table,rowkey_col,namespace='recsys',column_family='cf'):
    host = 'host1:2181，host2:2181,host3:2181,host4:2181,host5:2181'  
    table = "{nm}:{tb}".format(nm=namespace,tb=hbase_table)  
    keyConv = "org.apache.spark.examples.pythonconverters.StringToImmutableBytesWritableConverter"  
    valueConv = "org.apache.spark.examples.pythonconverters.StringListToPutConverter"  
    conf = {"hbase.zookeeper.quorum": host,  
        "hbase.mapred.outputtable": table,  
        "mapreduce.outputformat.class": "org.apache.hadoop.hbase.mapreduce.TableOutputFormat",  
        "mapreduce.job.output.key.class": "org.apache.hadoop.hbase.io.ImmutableBytesWritable",  
        "mapreduce.job.output.value.class": "org.apache.hadoop.io.Writable"}  
    columns = df.columns
    rdd = df.rdd.flatMap(lambda r : flatRow(r,rowkey_col,column_family,columns))
    rdd.saveAsNewAPIHadoopDataset(conf=conf,keyConverter=keyConv,valueConverter=valueConv) 

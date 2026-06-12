# SMBus Application API Usage Examples

## Introduction
The SMBus driver allows application software to send and receive SMBus commands using the SMBus FPGA IP block.
This can be done using the supplied API functions listed in the smbus.h file.
The driver will allow up to 7 SMBus instances to be created. All of these instances can be used as both as an SMBus target or controller.
All the instances will behave as if they are separate devices on the SMBus.
The instances can be initialized as fixed addresses or be ARP capable and can be initialized to support PEC or not.

The SMBus driver operates as an event driven state machine. The events are generated from interrupts triggered from the the SMBus hardware IP block.

It is essential that the correct interrupt is attached by application code and the SMBus driver's interrupt handler called when the interrupt is fired. 

Detailed examples of how to use the driver APIs can be found below:

## Memory Assignment

The SMBus driver will create a copy of the profile it uses internally and pass a pointer to that to application from the xInitSMBus().
Application will be required to store the pointer for accessing further SMBus APIs.

```sh
/* Application code will be passed a pointer to the SMBus profile from xInitSMBus() */  
struct  SMBUS_PROFILE_TYPE* pxSMBusProfile = NULL;
```



## Interrupt Code

Application code will attach the correct hardware interrupt to the interrupt servicing code and associate the SMBus driver's handler vSMBusInterruptHandler with this interrupt

```sh
/* Application code will attach the correct hardware interrupt to the interrupt servicing code and associate the SMBus driver's handler vSMBusInterruptHandler with this interrupt */

/* Example for an ARM core. For Microblaze a different function would be required */

#define INTC_INTERRUPT      XPAR_FABRIC_BLP_BLP_LOGIC_AXI_SMBUS_RPU_IP2INTC_IRPT_INTR       /* This value may be found in the xparameters.h file */
#define INTC                XScuGic
#define INTC_HANDLER        XScuGic_InterruptHandler
#define INTC_DEVICE_ID      XPAR_SCUGIC_0_DEVICE_ID  /* = 0 */


static int ulSMBusSetupInterruptSystem( )
{
    int result = XST_FAILURE;
    XScuGic *pxIntcInstancePtr = &Intc;
    XScuGic_Config *pxIntcConfig = NULL;
    pxIntcConfig = XScuGic_LookupConfig( INTC_DEVICE_ID );
    if( pxIntcConfig == NULL )
    {
        return XST_FAILURE;
    }
    
    result = XScuGic_CfgInitialize( pxIntcInstancePtr, pxIntcConfig, pxIntcConfig->CpuBaseAddress );
    if( result != XST_SUCCESS )
    {
        return XST_FAILURE;
    }
    
    /* Connect the interrupt handler */
    result = XScuGic_Connect( pxIntcInstancePtr, INTC_INTERRUPT,( Xil_ExceptionHandler )vSMBusInterruptHandler, pxSMBusProfile );
    if( result != XST_SUCCESS ) 
    {
        return result;
    }
    
    /* Enable the interrupt for the IP */
    XScuGic_Enable( pxIntcInstancePtr, INTC_INTERRUPT );
    Xil_ExceptionInit( );
    Xil_ExceptionRegisterHandler( XIL_EXCEPTION_ID_INT,
       ( Xil_ExceptionHandler )INTC_HANDLER, pxIntcInstancePtr );
    Xil_ExceptionEnable( );    /* Enable non-critical exceptions */
    return XST_SUCCESS;
}
```






## Callback Functions
Several callback functions need to be implemented by application software and passed to the driver on creation of each instance.
NOTE: The callback functions may be called from within an interrupt context so it is important that they complete quickly




### Given Command Get Protocol Callback

This callback will be called when an SMBus Target has received a command byte from an SMBus Controller.
In order for the driver to know how to interpret the bytes following the command it needs to know what SMBus protocol
the command uses.

```sh
/* This callback function must be created. It will map a received command byte to an SMBus Protocol
     This allows the driver to understand how to interpret the SMBus message */
void  GivenCommandGetProtocolCallback( uint8_t ucCommand, SMBus_Command_Protocol_Type* pxProtocol )
{
    if( NULL != pxProtocol )
    {
        switch( ucCommand )
        {
        case 5:
            *pxProtocol = SMBUS_PROTOCOL_WRITE_BYTE;
            break;
        case 6:
            *pxProtocol = SMBUS_PROTOCOL_WRITE_WORD;
            break;
        case 7:
            *pxProtocol = SMBUS_PROTOCOL_READ_BYTE;
            break;
        case 8:
            *pxProtocol = SMBUS_PROTOCOL_READ_WORD;
            break;
        case 9:
            *pxProtocol = SMBUS_PROTOCOL_PROCESS_CALL;
            break;
        case 10:
            *pxProtocol = SMBUS_PROTOCOL_BLOCK_WRITE;
            break;
        case 11:
            *pxProtocol = SMBUS_PROTOCOL_BLOCK_READ;
            break;
        case 12:
            *pxProtocol = SMBUS_PROTOCOL_BLOCK_WRITE_BLOCK_READ_PROCESS_CALL;
            break;
        case 13:
            *pxProtocol = SMBUS_PROTOCOL_HOST_NOTIFY;
            break;
        case 14:
            *pxProtocol = SMBUS_PROTOCOL_WRITE_32;
            break;
        case 15:
            *pxProtocol = SMBUS_PROTOCOL_READ_32;
            break;
        case 16:
            *pxProtocol = SMBUS_PROTOCOL_WRITE_64;
            break;
        case 17:
            *pxProtocol = SMBUS_PROTOCOL_READ_64;
            break;

        default:
            *pxProtocol = SMBUS_PROTOCOL_NONE;
            break;
        }
    }
}
```



### Read Data Callback
This callback will be called when an SMBus Target has received a read protocol message from an SMBus Controller.
The target must have the data immediately available.
Note - the pusDataSize will be overwritten with the correct values for RECEIVE_BYTE (1), READ_BYTE (1), READ_WORD (2), READ_32 (4), READ_64 (8), and PROCESS_CALL (2), regardless of what value is set in the callback.

```sh
/* This callback function must be created. It will be called when an SMBus target is required to return data 
     for any read protocol */
void  ReadDataCallback( uint8_t ucCommand, uint8_t* pucData, uint16_t* pusDataSize )
{
    int i = 0;
    
    if( ( NULL != pusDataSize ) &&
        ( NULL != pucData ) )
    {
        switch( ucCommand )
        {

        case 7:
            /* Byte Read */           
            *pusDataSize = 1;
            pucData[0] = ReadByteData;
            break;

        case 8:   
            /* Word Read */             
            *pusDataSize = 2;
            pucData[0] = ReadWordData[0];
            pucData[1] = ReadWordData[1];
            break;


        case 9:   
            /* Process Call */          
            *pusDataSize = 2;
            pucData[0] = ProcessCallReadData[0];
            pucData[1] = ProcessCallReadData[1];;
            break;

        case 11:      
            /* Block Read */
            /* +1 as we need to count the block size byte also */
            *pusDataSize = BlockReadSize+1; 

            pucData[0] = BlockReadSize;
            for( i = 1; i <= BlockReadSize; i++ )
            {
                pucData[i] = BlockReadData[i];
            }
            break;

        case 12:               
            /* Block Write Block Read Process Call */
            /* +1 as we need to count the block size byte also */
            *pusDataSize = BlockWriteBlockReadSize+1; 

            pucData[0] = BlockWriteBlockReadSize;
            for( i = 1; i <= BlockWriteBlockReadSize; i++ )
            {
                pucData[i] = BlockWriteBlockReadData[i];

            break;

        case 15:
            /* Read 32 */         
            *pusDataSize = 4;
            pucData[0] = Read32Data[0];
            pucData[1] = Read32Data[1];
            pucData[2] = Read32Data[2];
            pucData[3] = Read32Data[3];
            break;

        case 17:                
            /* Read 64 */
            *pusDataSize = 8;
            pucData[0] = Read64Data[0];
            pucData[1] = Read64Data[1];
            pucData[2] = Read64Data[2];
            pucData[3] = Read64Data[3];
            pucData[4] = Read64Data[4];
            pucData[5] = Read64Data[5];
            pucData[6] = Read64Data[6];
            pucData[7] = Read64Data[7];
            break;

        default:
            *pusDataSize = 0;
            break;
        }
    }
}
```

### Write Data Callback
This callback will be called when an SMBus Target has received data from an SMBus Controller as part of one of the write protocol messages or
it will be called by an SMBus Controller when it has received data back from an SMBus Target as part of one of the read protocol messages.

```sh
/* This callback function must be created. It will be called when an SMBus target has been sent data 
as part of any write protocol */
void  WriteDataCallback( uint8_t ucCommand, uint8_t* pucData, uint16_t usDataSize )
{
    ucTargetCommand = ucCommand;
    usTargetDataSize = usDataSize;
    
    if( ( SMBUS_DATA_SIZE_MAX >= usDataSize ) &&
        ( NULL != pucData ) )
    {
        memcpy( ucTargetData, pucData, usDataSize );
        ucTargetWriteCallback = SMBUS_TRUE;
    }
}
```



### Announce Result Callback
This callback will be called when an SMBus transaction completes.
If its an SMBus transaction on an SMBus Target the Transaction ID will be 0.
If its an SMBus transaction initiated on an SMBus Controller then the Transaction ID will be the ID returned by the xSMBusControllerInitiateCommand().

```sh
/* This callback function must be created. It will be called when an SMBus transaction has finished  */
void AnnounceResultCallback(uint8_t ucCommand, uint32_t ulTransactionID, uint32_t ulResult)
{
    /* Application software may act on SMBus transaction completion */
    SMBusTransactionComplete(ucCommand, ulTransactionID, ulResult);
}
```






### ARP Address Change Callback
```sh
/* This callback function must be created for all ARP capable instances.
It will be called when an ARP Assign Address message has been sent to the instance   */
void AnnounceARPAddressChangeCallback(uint8_t ucNewAddress)
{
    StoreInstaneAddress(ucNewAddress );
    ucArpAddressChangeCallback = SMBUS_TRUE;

}
```



### Error Callback
```sh
/* This callback is optional. If created it will be called if an error interrupt has been received */
void AnnounceErrorCallback(uint8_t ucCommand, uint8_t  ucError)
{
    ActOnSMBusError(ucError);
 }
 ```



### Warning Callback
```sh
/* This callback is optional. If created it will be called if a warning interrupt has been received */
void AnnounceWarningCallback(uint8_t ucCommand, uint8_t  ucWarning)
{
    ActOnSMBusWarning(ucWarning);
}
```

## SMBus Driver Initialization and Instance Creation Example Code

```sh
/* Example application code to intialize the SMBus Driver, create a single SMBus instance and attach and enable interrupts to drive the SMBus driver */
 
uint8_t ucTargetUDID0[16]       = { 0x10, 0xFF, 0xFF, 0xFF, 0xF0, 0x0F, 0x0F, 0xF0, 0x07, 0x00, 0xFF, 0xFF, 0xEE, 0x10, 0x0F, 0x40 }; //ARP CAPABLE
 
int main( void )  
{
    SMBus_Error_Type                                    xReturnCode                 = SMBUS_ERROR;
    uint8_t                                             ucSMBusAddress              = 0;
    uint8_t                                             ucUDID[SMBUS_UDID_LENGTH]   = { 0 };
    SMBus_ARP_Capability                                xARPCapability              = SMBUS_ARP_CAPABILITY_UNKNOWN;
    SMBUS_USER_SUPPLIED_ENVIRONMENT_GET_PROTOCOL_TYPE   pFnGetProtocol              = NULL;
    SMBUS_USER_SUPPLIED_ENVIRONMENT_GET_DATA_TYPE       pFnGetData                  = NULL;
    SMBUS_USER_SUPPLIED_ENVIRONMENT_WRITE_DATA_TYPE     pFnWriteData                = NULL;
    SMBUS_USER_SUPPLIED_ENVIRONMENT_COMMAND_COMPLETE    pFnAnnounceResult           = NULL;
    SMBUS_USER_SUPPLIED_ENVIRONMENT_ARP_ADRRESS_CHANGE  pFnArpAddressChange         = NULL;
    SMBUS_USER_SUPPLIED_ENVIRONMENT_BUS_ERROR           pFnBusError                 = NULL;
    SMBUS_USER_SUPPLIED_ENVIRONMENT_BUS_WARNING         pFnBusWarning               = NULL;
    uint8_t                                             ucSimpleDevice              = 0;
 
   /*   Initialize the SMBus Driver passing in: 
        a pointer to it's memory profile, 
        the frequency class to be used 100KHz, 400KHz or 1MHz, 
        the base address of the SMBus IP's register set, found in xparameters.h
        the level of log detail required and
        <OPTIONAL> a pointer to a get tick count function for logs */
        
    xReturnCode = xInitSMBus( &pxSMBusProfile, SMBUS_FREQ_1MHZ,( void* )XPAR_SMBUS_0_BASEADDR, SMBUS_LOG_LEVEL_DEBUG, 
                                    ( SMBUS_USER_SUPPLIED_ENVIRONMENT_READ_TICKS )&vGetTicksFromApplication );
     
    if( SMBUS_SUCCESS == xReturnCode )
    {
        /* Get ready to create an SMBus Instance 
           Several parameters are required to be set before an instance can be created */
         
        /*  ARP Cabability: The SMBus instance can be created with one of 4 ARP capabilities:
                                         SMBUS_ARP_CAPABLE,
                                         SMBUS_ARP_FIXED_AND_DISCOVERABLE,
                                         SMBUS_ARP_FIXED_NOT_DISCOVERABLE,
                                         SMBUS_ARP_NON_ARP_CAPABLE
                                         The required capability must be set on creation of the instance */
        xARPCapability = SMBUS_ARP_CAPABLE;
         
        /* SMBus Address:   If the Instance is to be ARP capable then no address is required otherwise 
                                           a fixed address must be set */     
        ucSMBusAddress = 0x2A;
         
        /* SMBus UDID:  A 16 byte UDID is required for the SMBus Instance and must be set on creation 
                        THe UDID will also determine if this SMBus instance can accept PEC */ 
        for( i = 0; i < SMBUS_UDID_LENGTH; i++ )
        {
            ucUDID[i] = ucTargetUDID0[i];
        }
         
        /* SMBus Callback Functions:    Separate callback functions for each instance must be created in the application software and 
                                        attached to the instance on creation 
                                        See above for examples of these callback functions */
        pFnGetProtocol           =( SMBUS_USER_SUPPLIED_ENVIRONMENT_GET_PROTOCOL_TYPE )&GivenCommandGetProtocolCallback;
        pFnGetData               =( SMBUS_USER_SUPPLIED_ENVIRONMENT_GET_DATA_TYPE )&ReadDataCallback;
        pFnWriteData             =( SMBUS_USER_SUPPLIED_ENVIRONMENT_WRITE_DATA_TYPE )&WriteDataCallback;
        pFnAnnounceResult        =( SMBUS_USER_SUPPLIED_ENVIRONMENT_COMMAND_COMPLETE )&AnnounceResultCallback;
        pFnArpAddressChange      =( SMBUS_USER_SUPPLIED_ENVIRONMENT_ARP_ADRRESS_CHANGE )&AnnounceARPAddressChangeCallback;
        pFnBusError              =( SMBUS_USER_SUPPLIED_ENVIRONMENT_BUS_ERROR )&AnnounceErrorCallback;
        pFnBusWarning            =( SMBUS_USER_SUPPLIED_ENVIRONMENT_BUS_WARNING )&AnnounceWarningCallback;
         
        /* SMBus Simple Device:     An instance can be created as a Simple Device. 
                                    If ucSimpleDevice = 1 the instance will ONLY accept SEND BYTE and RECEIVE BYTE protocols */
        ucSimpleDevice = 0;                            
                                     
        /* Now create the instance, the instance number 0 - 6 will be returned */
        uint8_t ucInstanceId            = ucCreateSMBusInstance( pxSMBusProfile, ucSMBusAddress, ucUDID, xARPCapability,
                                                                    pFnGetProtocol, pFnGetData, pFnWriteData, pFnAnnounceResult,
                                                                    pFnArpAddressChange, pFnBusError, pFnBusWarning, ucSimpleDevice );
 
        /*  NOTE: The ucCreateSMBusInstance() function can be called to create up to 7 unique SMBus Instances
            each with a different ARP capabilities, SMBUs Address and UDID by repeating the steps in lines 25 - 63 above */
     
        /* Disable and clear all SMBus interrupts */
        if(SMBUS_SUCCESS == xSMBusInterruptDisableAndClearInterrupts( pxSMBusProfile ))
        {
            /*  Attach the SMBus hardware interrupt to the interrupt system and
                associate the SMBus Interrupt Handler function with the interrupt */
            if( XST_SUCCESS == ulSMBusSetupInterruptSystem( ) )
            {
                /* Enable SMBus Interrupts */
                if(SMBUS_SUCCESS == xSMBusInterruptEnableInterrupts( pxSMBusProfile ))
                {
                    /* SMBus driver is now ready to accept SMBus messages */
                }
            }  
        }
    }
}
```

## Operation as an SMBus Controller
Any instance that has been created can be used to initiate an SMBus transaction as a controller.
The application software needs simply to call the xSMBusControllerInitiateCommand() with the necessary parameters.

An example of a Write64 is shown below
```sh
    uint8_t                     ucSMBusInstance                     = 0;
    uint8_t                     ucTargetAddress                     = 0;
    uint8_t                     ucCommand                           = 0; 
    SMBus_Command_Protocol_Type xProtocol                           = SMBUS_PROTOCOL_NONE;
    uint8_t                     ucPECRequired                       = SMBUS_FALSE;
    uint32_t                    ulTransactionID                     = 0;
    uint8_t                     ucControllerDataToSend[MAX_DATA]    = { 0 };
    uint16_t                    usControllerDataToSendSize          = 0;
    uint8_t                     ucBlockSize                         = 0;
    uint8_t                     ucInitialValue                      = 0; 

    /* Assign values to all the parameters */
    ucSMBusInstance             = 0;                                    /* This will be a value 0 - 6 which will have been returned by ucCreateSMBusInstance() function */
    ucTargetAddress             = 0x6A;                                 /* The address of the the SMBus Target the message is to be sent to */
    ucCommand                   = 16;                                   /* The command byte to be sent. The SMBus Target must know what protocol this command maps to */
    xProtocol                   = SMBUS_PROTOCOL_WRITE_64;              /* The protocol to be used for this SMBus message */
    ucPECRequired               = SMBUS_FALSE;                          /* Is the SMBus Target expecting a PEC byte to be sent */

    /* Write Data of 8 bytes for a Write64 Protocol */
    ucControllerDataToSend[0]   = 29;
    ucControllerDataToSend[1]   = 250;
    ucControllerDataToSend[2]   = 35;
    ucControllerDataToSend[3]   = 14;
    ucControllerDataToSend[4]   = 55;
    ucControllerDataToSend[5]   = 86;
    ucControllerDataToSend[6]   = 127;
    ucControllerDataToSend[7]   = 48;

    ucControllerDataToSendSize  = 8;

    if( SMBUS_SUCCESS == xSMBusControllerInitiateCommand( pxSMBusProfile,  ucSMBusInstance, ucTargetAddress, ucCommand, xProtocol,
                                                            ucControllerDataToSendSize, ucControllerDataToSend, ucPECRequired, &ulTransactionID ) )
    {
        /* Write 64 Initiated */
        /* This function may return before the asynchronous transaction has completed */
    }
```        




An example of a Block Write - Block Read Process Call is shown below
```sh
    uint8_t                     ucSMBusInstance                     = 0;
    uint8_t                     ucTargetAddress                     = 0;
    uint8_t                     ucCommand                           = 0; 
    SMBus_Command_Protocol_Type xProtocol                           = SMBUS_PROTOCOL_NONE;
    uint8_t                     ucPECRequired                       = SMBUS_FALSE;
    uint32_t                    ulTransactionID                     = 0;
    int                         i                                   = 0;
    uint8_t                     ucControllerDataToSend[MAX_DATA]    = { 0 };
    uint16_t                    usControllerDataToSendSize          = 0;
    uint8_t                     ucBlockSize                         = 0;
    uint8_t                     ucInitialValue                      = 0;

    /* Assign values to all the parameters */
    ucSMBusInstance = 0;                                                        /* This will be a value 0 - 6 which will have been returned by ucCreateSMBusInstance() function */
    ucTargetAddress = 0x6A;                                                     /* The address of the the SMBus Target the message is to be sent to */
    ucCommand       = 12;                                                       /* The command byte to be sent. The SMBus Target must know what protocol this command maps to */
    xProtocol       = SMBUS_PROTOCOL_BLOCK_WRITE_BLOCK_READ_PROCESS_CALL;       /* The protocol to be used for this SMBus message */
    ucPECRequired   = false;                                                    /* Is the SMBus Target expecting a PEC byte to be sent */


    /* Write Data  -    For block transactions the block size needs to be added as the first byte of the data supplied to the function 
                        So for a block size of 125, 126 bytes will be supplied to the function, the first byte being the value 125 and the
                        SendSize parameter will be 126 */
    ucBlockSize = 125;
    ucControllerDataToSend[0] =  ucBlockSize;  
    ucControllerDataToSend[1] = 0;
    ucInitialValue = 0x1;
    for( i = 1; i <= ucBlockSize; i++ )
    {
       ucControllerDataToSend[i] = ucInitialValue++;
    }
    ucControllerDataToSendSize =  ucBlockSize + 1;
           
     if( SMBUS_SUCCESS == xSMBusControllerInitiateCommand( pxSMBusProfile,  ucSMBusInstance, ucTargetAddress, ucCommand, xProtocol, 
                                                            ucControllerDataToSendSize, ucControllerDataToSend, ucPECRequired, &ulTransactionID ) )
    {
        /* Block Write - Block Read Process Call Initiated */
        /* This function may return before the asynchronous transaction has completed */
    }
```                  

## ARP Operation

If any of the SMBus Instances have been created as ARP capable then they will respond to any ARP command sent from an ARP Controller to the SMBus Device Default Address .

ARP commands include:
Prepare to ARP
Reset device (general)
Get UDID (general)
Assign address
Get UDID (directed)
Reset device ARP (directed)
Notify ARP Controller

Only after an ARP capable instance has been given an address by an ARP Assign Address command will it be ready to receive any standard SMBus commands.
On receiving an ARP Assign Address command the instance receiving the change will call it's ARP Address Change Callback function to notify the application of the assigned address.

(Note: Unlike ARP capable instances, fixed address instances are ready to receive any standard SMBus commands as soon as they are created.)

 

## Logging
Event logs are written to a 5000 entry deep circular buffer.
Various levels of logging can be set ranging from SMBUS_LOG_LEVEL_NONE to SMBUS_LOG_LEVEL_DEBUG level logging.
Logs can be retreived using the xSMBusGetLog() function which automatically formats the log events into a text string.
Logs can be cleared using the xSMBusLogReset() function.

```sh
void vPostTestPrintLog( void )
{
    char        cLogBuffer[LOGSIZE]     = { 0 };
    uint32_t    ulLogSize               = 0;
   /**********************************  LOG    **********************************/  
    xil_printf( "Log\n\r" );
    cLogBuffer[0] = '\0';           
    ulLogSize = 0;
    xSMBusGetLog( pxSMBusProfile, cLogBuffer, &ulLogSize );
    xil_printf( "%s\n\r", cLogBuffer );

}
```      



An example log for an SMBus Controller (Instance 0) sending a  Write64 command to SMBus Target (Instance 1 ) is shown below:
```sh
0000 3793799 HW_WRITE  88 0x00000a08 0x80000000
0001 3793799 HW_WRITE  88 0x00000a10 0x80000000
0002 3793799 HW_WRITE  88 0x00000020 0x00000000
0003 3793799 FSM        0 SMBUS_STATE_INITIAL E_SEND_NEXT_BYTE
0004 3793799 DEBUG      0 0x00000024 line 3623
0005 3793800 HW_READ   88 0x00000830 0x00000001
0006 3793800 DEBUG      0 0x00000001 line 662
0007 3793800 HW_READ   88 0x00000a1c 0x00000001
0008 3793800 DEBUG      0 0x00000001 line 665
0009 3793800 DEBUG      0 0x0000006a line 701
0010 3793800 HW_READ   88 0x00000a0c 0x00000003
0011 3793800 HW_WRITE  88 0x00000a08 0x000000d4
0012 3793800 FSM        0 SMBUS_STATE_CONTROLLER_SEND_COMMAND E_SEND_NEXT_BYTE
0013 3793800 DEBUG      0 0x00000024 line 3623
0014 3793801 HW_READ   88 0x00000a0c 0x00000102
0015 3793801 HW_WRITE  88 0x00000a08 0x00000210
0016 3793801 FSM        0 SMBUS_STATE_CONTROLLER_WRITE_BYTE E_SEND_NEXT_BYTE
0017 3793801 DEBUG      0 0x00000024 line 3623
0018 3793801 DEBUG      0 0x00000008 line 2615
0019 3793801 HW_READ   88 0x00000a0c 0x00000200
0020 3793801 HW_WRITE  88 0x00000a08 0x0000021d
0021 3793801 HW_WRITE  88 0x00000a08 0x000002fa
0022 3793801 HW_WRITE  88 0x00000a08 0x00000223
0023 3793802 HW_WRITE  88 0x00000a08 0x0000020e
0024 3793802 HW_WRITE  88 0x00000a08 0x00000237
0025 3793802 HW_WRITE  88 0x00000a08 0x00000256
0026 3793802 HW_WRITE  88 0x00000a08 0x0000027f
0027 3793802 HW_WRITE  88 0x00000a00 0x00000001
0028 3793803 HW_WRITE  88 0x00000024 0x0000dfef
0029 3793803 HW_WRITE  88 0x00000020 0x00000001
0030 3793803 HW_WRITE  88 0x00000020 0x00000000
0031 3793803 HW_READ   88 0x00000028 0x00000080
0032 3793804 HW_READ   88 0x00000024 0x0000dfef
0033 3793804 INTERRUPT 88 0x00067f94 0x00000080
0034 3793804 HW_READ   88 0x00000600 0x000001d4
0035 3793804 FSM        1 SMBUS_STATE_INITIAL E_TARGET_WRITE_IRQ
0036 3793804 DEBUG      1 0x00000001 line 3623
0037 3793804 HW_WRITE  88 0x00000030 0x00000000
0038 3793805 HW_WRITE  88 0x00000028 0x00000080
0039 3793805 HW_WRITE  88 0x00000020 0x00000001
0040 3793805 HW_WRITE  88 0x00000020 0x00000000
0041 3793805 HW_READ   88 0x00000028 0x00000130
0042 3793805 HW_READ   88 0x00000024 0x0000dfef
0043 3793805 INTERRUPT 88 0x00067f94 0x00000130
0044 3793805 FSM        1 SMBUS_STATE_AWAITING_COMMAND_BYTE E_TARGET_DATA_IRQ
0045 3793805 DEBUG      1 0x00000003 line 3623
0046 3793805 DEBUG      1 0x0000006a line 831
0047 3793806 HW_READ   88 0x00000610 0x00040102
0048 3793806 HW_READ   88 0x0000060c 0x00000010
0049 3793806 PROTOCOL   1 0x00000010 SMBUS_PROTOCOL_WRITE_64
0050 3793806 HW_WRITE  88 0x00000614 0x00000008
0051 3793806 HW_READ   88 0x00000608 0x00000003
0052 3793806 HW_WRITE  88 0x00000604 0x00000000
0053 3793806 HW_READ   88 0x00000608 0x00000003
0054 3793806 HW_WRITE  88 0x00000604 0x00000000
0055 3793807 HW_READ   88 0x00000608 0x00000102
0056 3793807 HW_WRITE  88 0x00000604 0x00000000
0057 3793807 HW_READ   88 0x00000608 0x00000200
0058 3793807 HW_WRITE  88 0x00000604 0x00000000
0059 3793807 HW_READ   88 0x00000608 0x00000300
0060 3793807 HW_WRITE  88 0x00000604 0x00000000
0061 3793807 HW_READ   88 0x00000608 0x00000300
0062 3793807 HW_WRITE  88 0x00000604 0x00000000
0063 3793807 HW_READ   88 0x00000608 0x00000400
0064 3793807 HW_WRITE  88 0x00000604 0x00000000
0065 3793808 HW_READ   88 0x00000608 0x00000500
0066 3793808 HW_WRITE  88 0x00000604 0x00000000
0067 3793808 HW_READ   88 0x00000608 0x00000600
0068 3793808 HW_WRITE  88 0x00000604 0x00000000
0069 3793808 HW_READ   88 0x00000610 0x00040102
0070 3793808 FSM        1 SMBUS_STATE_AWAITING_DATA E_DESC_FIFO_ALMOST_EMPTY_IRQ
0071 3793808 DEBUG      1 0x00000026 line 3623
0072 3793808 HW_READ   88 0x00000608 0x00000600
0073 3793809 FSM        1 SMBUS_STATE_AWAITING_DATA E_TARGET_DATA_IRQ
0074 3793809 DEBUG      1 0x00000003 line 3623
0075 3793809 HW_READ   88 0x00000610 0x00040200
0076 3793809 HW_READ   88 0x0000060c 0x0000001d
0077 3793809 HW_READ   88 0x0000060c 0x000000fa
0078 3793809 HW_WRITE  88 0x00000614 0x00000001
0079 3793809 HW_WRITE  88 0x00000030 0x00000000
0080 3793809 HW_WRITE  88 0x00000028 0x00000130
0081 3793810 HW_WRITE  88 0x00000020 0x00000001
0082 3793810 HW_WRITE  88 0x00000020 0x00000000
0083 3793810 HW_READ   88 0x00000028 0x00000020
0084 3793810 HW_READ   88 0x00000024 0x0000dfef
0085 3793810 INTERRUPT 88 0x00067f94 0x00000020
0086 3793810 FSM        1 SMBUS_STATE_AWAITING_DATA E_TARGET_DATA_IRQ
0087 3793810 DEBUG      1 0x00000003 line 3623
0088 3793810 HW_READ   88 0x00000610 0x00040200
0089 3793810 HW_READ   88 0x0000060c 0x00000023
0090 3793811 HW_READ   88 0x0000060c 0x0000000e
0091 3793811 HW_WRITE  88 0x00000614 0x00000001
0092 3793811 HW_WRITE  88 0x00000030 0x00000000
0093 3793811 HW_WRITE  88 0x00000028 0x00000020
0094 3793811 HW_WRITE  88 0x00000020 0x00000001
0095 3793811 HW_WRITE  88 0x00000020 0x00000000
0096 3793811 HW_READ   88 0x00000028 0x00000020
0097 3793811 HW_READ   88 0x00000024 0x0000dfef
0098 3793811 INTERRUPT 88 0x00067f94 0x00000020
0099 3793812 FSM        1 SMBUS_STATE_AWAITING_DATA E_TARGET_DATA_IRQ
0100 3793812 DEBUG      1 0x00000003 line 3623
0101 3793812 HW_READ   88 0x00000610 0x00040102
0102 3793812 HW_READ   88 0x0000060c 0x00000037
0103 3793812 HW_WRITE  88 0x00000614 0x00000001
0104 3793812 HW_WRITE  88 0x00000030 0x00000000
0105 3793812 HW_WRITE  88 0x00000028 0x00000020
0106 3793812 HW_WRITE  88 0x00000020 0x00000001
0107 3793812 HW_WRITE  88 0x00000020 0x00000000
0108 3793813 HW_READ   88 0x00000028 0x00008020
0109 3793813 HW_READ   88 0x00000024 0x0000dfef
0110 3793813 INTERRUPT 88 0x00067f94 0x00008020
0111 3793813 FSM        1 SMBUS_STATE_AWAITING_DATA E_TARGET_DATA_IRQ
0112 3793813 DEBUG      1 0x00000003 line 3623
0113 3793813 HW_READ   88 0x00000610 0x00040200
0114 3793813 HW_READ   88 0x0000060c 0x00000056
0115 3793813 HW_READ   88 0x0000060c 0x0000007f
0116 3793813 HW_WRITE  88 0x00000614 0x00000001
0117 3793814 FSM        0 SMBUS_STATE_CONTROLLER_WRITE_BYTE E_CONTROLLER_DESC_FIFO_ALMOST_EMPTY_IRQ
0118 3793814 DEBUG      0 0x00000017 line 3623
0119 3793814 DEBUG      0 0x00000001 line 2615
0120 3793814 DEBUG      0 0x00000001 line 2643
0121 3793814 DEBUG      0 0x00000001 line 2664
0122 3793814 DEBUG      0 0x00000000 line 2683
0123 3793814 HW_READ   88 0x00000a0c 0x00000003
0124 3793814 HW_WRITE  88 0x00000a08 0x00000330
0125 3793814 HW_WRITE  88 0x00000a00 0x00000001
0126 3793815 HW_WRITE  88 0x00000030 0x00000000
0127 3793815 HW_WRITE  88 0x00000028 0x00008020
0128 3793815 HW_WRITE  88 0x00000020 0x00000001
0129 3793815 HW_WRITE  88 0x00000020 0x00000000
0130 3793815 HW_READ   88 0x00000028 0x00002120
0131 3793815 HW_READ   88 0x00000024 0x0000dfef
0132 3793815 INTERRUPT 88 0x00067f94 0x00002120
0133 3793816 FSM        1 SMBUS_STATE_AWAITING_DATA E_TARGET_DATA_IRQ
0134 3793816 DEBUG      1 0x00000003 line 3623
0135 3793816 HW_READ   88 0x00000610 0x00040102
0136 3793816 HW_READ   88 0x0000060c 0x00000030
0137 3793816 HW_WRITE  88 0x00000614 0x00000001
0138 3793816 FSM        1 SMBUS_STATE_AWAITING_DONE E_DESC_FIFO_ALMOST_EMPTY_IRQ
0139 3793816 DEBUG      1 0x00000026 line 3623
0140 3793816 DEBUG      1 0x00000009 line 1654
0141 3793816 DEBUG      1 0x00000008 line 1657
0142 3793816 DEBUG      1 0x00000008 line 1660
0143 3793817 HW_READ   88 0x00000608 0x00000003
0144 3793817 HW_WRITE  88 0x00000030 0x00000000
0145 3793817 HW_WRITE  88 0x00000028 0x00002120
0146 3793817 HW_WRITE  88 0x00000020 0x00000001
0147 3793817 HW_WRITE  88 0x00000020 0x00000000
0148 3793817 HW_READ   88 0x00000028 0x00001008
0149 3793818 HW_READ   88 0x00000024 0x0000dfef
0150 3793818 INTERRUPT 88 0x00067f94 0x00001008
0151 3793818 FSM        1 SMBUS_STATE_AWAITING_DONE E_TARGET_DONE_IRQ
0152 3793818 DEBUG      1 0x00000004 line 3623
0153 3793818 DEBUG      1 0x00000010 line 1773
0154 3793818 DEBUG      1 0x00000008 line 1776
0155 3793818 DEBUG      1 0x00000001 line 68
0156 3793818 HW_READ   88 0x00000610 0x00040003
0157 3793818 HW_WRITE  88 0x0000060c 0x80000000
0158 3793819 FSM        0 SMBUS_STATE_AWAITING_DONE E_CONTROLLER_DONE_IRQ
0159 3793819 DEBUG      0 0x00000016 line 3623
0160 3793819 DEBUG      0 0x00000000 line 68
0161 3793819 HW_READ   88 0x00000a14 0x00120003
0162 3793819 HW_WRITE  88 0x00000a10 0x80000000
0163 3793819 HW_WRITE  88 0x00000a08 0x80000000
0164 3793820 HW_WRITE  88 0x00000030 0x00000000
0165 3793820 HW_WRITE  88 0x00000028 0x00001008
0166 3793820 HW_WRITE  88 0x00000024 0x000001ef
0167 3793820 HW_WRITE  88 0x00000020 0x00000001
```    

## Simple Target
An SMBus instance may be created as a "Simple" target.
This can be done by setting xInstance.ucSimpleDevice = 1 when the creating the instance.
When this is done the instance will ONLY accept SEND BYTE and RECEIVE BYTE protocols.




## SMBus Address
The SMBus Address is a 7-bit value. When using the ucCreateSMBusInstance(), the value set in the ucSMBusAddress field of the Instance parameter should occupy the 7 least significant bits (bits 6 - 0).
When this address is transmitted on the SMBus it may be seen bit shifted to the left by a single bit. The 7 bits of the address now become the 7 most significant bits (bits 7 - 1) and bit 0 is used as the READ/WRITE bit.

When an ARP Controller assigns a new address via the ARP ASSIGN ADDRESS command, the new 7-bit assigned address must be added to the payload already pre-shifted to the 7 most significant bits (bits 7 - 1) and bit 0 is ignored.




## Dynamic and Persistent Addresses
If an SMBus instance is created as ARP-Capable and its address is Dynamic and Persistent  (ie. Address Type bits 7:6 of the Device Capabilities Field of the instance's UDID are set to be Dynamic and Persistent [01] )
In this situation:
ucCreateSMBusInstance() could be creating the instance with an invalid temporary address and awaiting a new address to be assigned by ARP
or
 ucCreateSMBusInstance() could be creating the instance with an address it had previously been assigned (ie a Persistent Address)

In order to determine which, the ucCreateSMBusInstance() function will check bit 7 of the ucSMBusAddress field in the Instance parameter.

 - If this is set it will treat the address as an invalid temporary address, will clear the AR flag and wait on a new address to be assigned by ARP Assign Address.
 - If the bit is clear,  it will treat the address as a dynamic and persistent address, will set the AR flag and allow add the address onto the bus.
 